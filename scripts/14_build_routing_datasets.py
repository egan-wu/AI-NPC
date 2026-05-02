"""
14_build_routing_datasets.py — Phase 2 LoRA Routing Dataset Builder

Produces two separate HuggingFace-format JSONL datasets from existing data:

  data/lora_inchar/   — pure in-character dialogue  (trains LoRA_InChar)
  data/lora_deflect/  — modern/jailbreak → in-character deflection  (trains LoRA_Deflect)

The two LoRA adapters are trained independently, then blended at runtime by
src/lora_router.py using the guard's continuous p value.

LoRA_InChar dataset:
  Source: existing mlx_mistral or mlx_curated_v9 data (already curated)
  Format: normal {system, user, assistant} triples — no prefix tokens

LoRA_Deflect dataset:
  Source: trap inputs (data/traps.jsonl) + deflection responses authored
          by a Haiku-class model in a follow-up step.
  Format: {system, user="<modern/jailbreak input>", assistant="<in-char deflect>"}
  Stubs with assistant=None are emitted to deflection_to_author.jsonl.

Usage:
    python scripts/14_build_routing_datasets.py \
        --inchar-src   data/mlx_mistral \
        --traps        data/traps.jsonl \
        --personas     configs/personas.yaml \
        --out-inchar   data/lora_inchar \
        --out-deflect  data/lora_deflect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Persona helper ────────────────────────────────────────────────────────────

def load_personas(yaml_path: Path) -> dict[str, str]:
    import yaml
    data = yaml.safe_load(yaml_path.read_text())
    return {p["id"]: p["system_prompt"] for p in data["personas"]}


# ── Soft-label strip (safety) ─────────────────────────────────────────────────

def strip_prefix_tokens(text: str) -> str:
    """
    Remove any soft-label tokens that might already be in the user content
    (in case src data was tagged by script 13). LoRA_InChar trains on raw
    inputs — no prefix tokens.
    """
    from src.soft_label import strip_token
    clean, _ = strip_token(text)
    return clean


# ── Build LoRA_InChar dataset ─────────────────────────────────────────────────

def build_inchar(src_dir: Path, out_dir: Path) -> None:
    print("Building LoRA_InChar dataset …")
    for split in ("train", "valid", "test"):
        src = src_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  ⚠️  {src} not found, skipping")
            continue
        rows = read_jsonl(src)
        cleaned = []
        for row in rows:
            msgs = []
            for m in row["messages"]:
                if m["role"] == "user":
                    msgs.append({"role": "user",
                                 "content": strip_prefix_tokens(m["content"])})
                else:
                    msgs.append(m)
            cleaned.append({"messages": msgs})
        write_jsonl(out_dir / f"{split}.jsonl", cleaned)
        print(f"  ✅ {split}: {len(cleaned)} rows → {out_dir / split}.jsonl")


# ── Build LoRA_Deflect stubs ───────────────────────────────────────────────────

def build_deflect_stubs(
    traps_path: Path,
    personas:   dict[str, str],
    out_dir:    Path,
) -> None:
    """
    For every trap, produce a JSONL stub with assistant=None.
    A follow-up Haiku authoring pass fills the deflection responses.

    Each row carries a _meta block describing:
      - trap_id, category (modern | jailbreak)
      - persona_id
      - deflect_style: the persona's characteristic deflection voice
    """
    print("Building LoRA_Deflect stubs …")

    DEFLECT_STYLE: dict[str, str] = {
        "innkeeper_marta":   "assume drunk or jesting; substitute an in-world equivalent",
        "merchant_garrick":  "pivot to a sale; claim 'eastern port' knowledge without explaining",
        "guard_roderick":    "treat as suspicious cant; demand they speak plainly or face arrest",
        "child_lily":        "respond with childlike confusion or wild wrong guess",
        "hermit_wenric":     "reframe as an omen or dream; pivot to philosophical reflection",
    }

    if not traps_path.exists():
        print(f"  ⚠️  {traps_path} not found — no stubs generated")
        return

    traps = read_jsonl(traps_path)
    stubs = []
    for t in traps:
        persona_id = t.get("persona_id", "innkeeper_marta")
        sys_prompt = personas.get(persona_id, "")
        if not sys_prompt:
            print(f"  ⚠️  unknown persona {persona_id!r}, skipping trap {t['id']}")
            continue
        stubs.append({
            "messages": [
                {"role": "system",    "content": sys_prompt},
                {"role": "user",      "content": t["user_prompt"]},
                {"role": "assistant", "content": None},   # to be filled by Haiku
            ],
            "_meta": {
                "trap_id":       t["id"],
                "category":      t["category"],
                "persona_id":    persona_id,
                "deflect_style": DEFLECT_STYLE.get(persona_id, "stay in character"),
                "needs_authoring": True,
            },
        })

    write_jsonl(out_dir / "deflection_to_author.jsonl", stubs)
    print(f"  ✅ {len(stubs)} deflection stubs → {out_dir}/deflection_to_author.jsonl")
    print(f"  ℹ️  Next: author responses in-session (Haiku), then run merge step below.\n")
    note = (
        "NEXT STEPS for LoRA_Deflect data:\n"
        "\n"
        "1. Switch active model to Haiku (per CLAUDE.md).\n"
        "2. For each stub in deflection_to_author.jsonl, author a one-sentence\n"
        "   in-character deflection that matches the deflect_style in _meta.\n"
        "3. Run scripts/14b_merge_deflect.py to produce train/valid/test splits.\n"
        "   (train 80%, valid 10%, test 10%)\n"
        "4. Proceed to RunPod training:\n"
        "     LoRA_InChar:  python scripts/05_train_hf.py --config configs/training_lora_inchar.yaml\n"
        "     LoRA_Deflect: python scripts/05_train_hf.py --config configs/training_lora_deflect.yaml\n"
    )
    (out_dir / "NEXT_STEPS.txt").write_text(note)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LoRA_InChar and LoRA_Deflect datasets for Phase 2 routing."
    )
    parser.add_argument("--inchar-src",  default="data/mlx_mistral",
                        help="Existing curated MLX/HF data dir for in-char samples.")
    parser.add_argument("--traps",       default="data/traps.jsonl")
    parser.add_argument("--personas",    default="configs/personas.yaml")
    parser.add_argument("--out-inchar",  default="data/lora_inchar")
    parser.add_argument("--out-deflect", default="data/lora_deflect")
    args = parser.parse_args()

    personas = load_personas(Path(args.personas))

    build_inchar(Path(args.inchar_src), Path(args.out_inchar))
    print()
    build_deflect_stubs(Path(args.traps), personas, Path(args.out_deflect))

    print("Done.")
    print(f"  LoRA_InChar  data → {args.out_inchar}/")
    print(f"  LoRA_Deflect stubs → {args.out_deflect}/deflection_to_author.jsonl")


if __name__ == "__main__":
    main()
