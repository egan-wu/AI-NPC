"""
15_inference_routing.py — Phase 2 Dual-LoRA Routing Inference & Analysis

Loads the trained LoRA_InChar + LoRA_Deflect adapters and runs them against:
  - 30 trap inputs (from data/traps.jsonl) covering modern, jailbreak, in-character
  - 10 normal in-world questions (hardcoded baselines)

For each input:
  1. Guard evaluates → (blocked, tier, tokens, p)
  2. LoRARouter blends adapters by p
  3. Response captured

Output:
  - outputs/routing_results.jsonl  (full records)
  - outputs/routing_summary.txt    (human-readable analysis table)

Usage:
    python scripts/15_inference_routing.py \
        --inchar   outputs/adapters/lora_inchar/best_adapter \
        --deflect  outputs/adapters/lora_deflect/best_adapter \
        --personas configs/personas.yaml \
        --traps    data/traps.jsonl \
        --out      outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

# Import directly from file to avoid src/__init__.py triggering npc_interface
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lora_router", _root / "src" / "lora_router.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LoRARouter = _mod.LoRARouter

# ── p-value mapping by trap category ─────────────────────────────────────────
# Guard is bypassed for clean evaluation; we assign p directly from category.
CATEGORY_P = {
    "baseline":     0.05,   # normal in-world dialogue → pure InChar
    "in_character": 0.10,   # normal questions, slightly cautious
    "modern":       0.80,   # anachronistic input → strong Deflect
    "jailbreak":    0.95,   # jailbreak attempt → maximum Deflect
}

# ── Extra normal-dialogue baselines (no trap, expected: pure in-character) ────
NORMAL_INPUTS = [
    ("innkeeper_marta",  "Do you have a room available tonight?"),
    ("innkeeper_marta",  "What's the news from the eastern road?"),
    ("merchant_garrick", "What's your best item today?"),
    ("merchant_garrick", "How far is the next town from here?"),
    ("guard_roderick",   "Is the gate open after dark?"),
    ("guard_roderick",   "Have you seen any suspicious strangers recently?"),
    ("child_lily",       "What do you do when you're scared?"),
    ("child_lily",       "Can you show me where the baker's house is?"),
    ("hermit_wenric",    "Why did you leave the Order?"),
    ("hermit_wenric",    "What do you know about the demon lord's origins?"),
]

BUCKET_LABELS = {
    (0.00, 0.25): "SAFE",
    (0.25, 0.45): "MILD",
    (0.45, 0.60): "MODERN_LOW",
    (0.60, 0.75): "MODERN_MID",
    (0.75, 0.90): "MODERN_HIGH",
    (0.90, 1.01): "JAILBREAK",
}


def p_to_bucket(p: float) -> str:
    if p < 0:
        return "KEYWORD_BLOCK"
    for (lo, hi), label in BUCKET_LABELS.items():
        if lo <= p < hi:
            return label
    return "JAILBREAK"


def load_personas(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {p["id"]: p["system_prompt"] for p in cfg["personas"]}


def load_traps(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_inference(
    router: LoRARouter,
    personas: dict,
    records: list[dict],
) -> list[dict]:
    results = []
    total = len(records)
    for i, rec in enumerate(records, 1):
        persona_id = rec["persona_id"]
        user_input = rec["user_input"]
        category   = rec.get("category", "baseline")
        system_prompt = personas[persona_id]

        # Assign p directly from category (clean controlled evaluation)
        p = CATEGORY_P.get(category, 0.5)
        bucket = p_to_bucket(p)

        print(f"[{i:2d}/{total}] [{category:<12}] p={p:.2f} | {persona_id[:18]} | {user_input[:50]}",
              flush=True)

        # Generate
        t1 = time.time()
        response = router.generate(
            system_prompt=system_prompt,
            user_text=user_input,
            p=p,
            max_new_tokens=180,
        )
        gen_ms = int((time.time() - t1) * 1000)

        results.append({
            **rec,
            "routing_p":    round(p, 4),
            "bucket":       bucket,
            "response":     response,
            "gen_ms":       gen_ms,
        })

        print(f"         → {response[:80]!r}  ({gen_ms}ms)", flush=True)
        print()

    return results


def write_summary(results: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("Phase 2 LoRA Routing — Inference Results")
    lines.append("=" * 90)
    lines.append("")

    # Group by category
    categories = {}
    for r in results:
        cat = r.get("category", "baseline")
        categories.setdefault(cat, []).append(r)

    for cat, recs in sorted(categories.items()):
        lines.append(f"── Category: {cat.upper()} ({len(recs)} samples) ──")
        lines.append(f"  {'ID':<25} {'Persona':<20} {'p':>5} {'Bucket':<14} {'Response (first 80 chars)'}")
        lines.append("  " + "-" * 85)
        for r in recs:
            resp_preview = r["response"].replace("\n", " ")[:80]
            lines.append(
                f"  {r['id']:<25} {r['persona_id']:<20} "
                f"{r['routing_p']:>5.2f} {r['bucket']:<14} {resp_preview!r}"
            )
        lines.append("")

    # Stats
    lines.append("── Statistics ──")
    lines.append(f"  Total samples     : {len(results)}")
    for cat in sorted(categories):
        lines.append(f"  Category [{cat}]: {len(categories[cat])} samples")

    # Bucket distribution
    from collections import Counter
    bucket_counts = Counter(r["bucket"] for r in results)
    lines.append("")
    lines.append("  Bucket distribution:")
    for bucket in ["SAFE", "MILD", "MODERN_LOW", "MODERN_MID", "MODERN_HIGH", "JAILBREAK", "KEYWORD_BLOCK"]:
        n = bucket_counts.get(bucket, 0)
        bar = "█" * n
        lines.append(f"    {bucket:<14} {n:3d}  {bar}")

    # Average p by category
    lines.append("")
    lines.append("  Mean routing_p by category:")
    for cat, recs in sorted(categories.items()):
        mean_p = sum(r["routing_p"] for r in recs) / len(recs)
        lines.append(f"    {cat:<20} mean_p = {mean_p:.3f}")

    # Timing
    mean_gen = sum(r["gen_ms"] for r in results) / len(results)
    lines.append("")
    lines.append(f"  Mean generation time: {mean_gen:.0f} ms/sample")
    lines.append("")
    lines.append("=" * 90)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inchar",   default="outputs/adapters/lora_inchar/best_adapter")
    parser.add_argument("--deflect",  default="outputs/adapters/lora_deflect/best_adapter")
    parser.add_argument("--personas", default="configs/personas.yaml")
    parser.add_argument("--traps",    default="data/traps.jsonl")
    parser.add_argument("--out",      default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading personas...")
    personas = load_personas(args.personas)

    print("Loading traps...")
    trap_recs = [
        {"id": t["id"], "category": t["category"],
         "persona_id": t["persona_id"], "user_input": t["user_prompt"]}
        for t in load_traps(args.traps)
    ]

    normal_recs = [
        {"id": f"normal_{i+1:02d}", "category": "baseline",
         "persona_id": pid, "user_input": uinput}
        for i, (pid, uinput) in enumerate(NORMAL_INPUTS)
    ]

    all_records = trap_recs + normal_recs
    print(f"Total inputs: {len(all_records)} ({len(trap_recs)} traps + {len(normal_recs)} baselines)")

    print("Loading LoRA Router (Mistral-7B + both adapters)...")
    router = LoRARouter(
        base_model_id="mistralai/Mistral-7B-Instruct-v0.3",
        inchar_path=args.inchar,
        deflect_path=args.deflect,
    )

    print("\n" + "=" * 70)
    print("Running inference...")
    print("p mapping: baseline=0.05 | in_character=0.10 | modern=0.80 | jailbreak=0.95")
    print("=" * 70 + "\n")

    results = run_inference(router, personas, all_records)

    # Save full JSONL
    jsonl_path = out_dir / "routing_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ Full results → {jsonl_path}")

    # Save summary
    summary_path = out_dir / "routing_summary.txt"
    write_summary(results, summary_path)
    print(f"✅ Summary → {summary_path}")


if __name__ == "__main__":
    main()
