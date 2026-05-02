"""
13_build_softlabel_dataset.py — Phase 1 Soft-Label Dataset Builder

Reads existing curated MLX data + held-out trap inputs, runs each input through
the trained guard to obtain p, maps p → bucket token, and prepends the token
to the user message.

Output structure (JSONL, MLX-compatible):
    {"messages":
        [{"role": "system",    "content": "..."},
         {"role": "user",      "content": "[SAFE] Where can I find lodging?"},
         {"role": "assistant", "content": "..."}]}

For trap inputs (modern / jailbreak), there is no in-character assistant
response in the original data. This script will emit them as
"deflection_needed" rows pointing to a separate file, where Haiku-class Claude
authors the in-character refusal in a follow-up step.

Usage:
    # Tag every existing in-char sample with [SAFE] and copy structure
    python scripts/13_build_softlabel_dataset.py \
        --in-data        data/mlx_curated_v9 \
        --traps          data/traps.jsonl \
        --out            data/mlx_softlabel \
        --personas       configs/personas.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.soft_label import prob_to_token, prepend_token, BUCKETS  # noqa: E402

# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Guard probability ─────────────────────────────────────────────────────────

def load_guard():
    """Lazy-load the trained guard from NPCInterface (re-uses the embedding
    layer that ships with src/npc_interface.py)."""
    from src.npc_interface import _GuardLayer  # noqa: WPS437  (intentional)
    return _GuardLayer()


# ── Per-row transformations ───────────────────────────────────────────────────

def _guard_prob(guard, text: str) -> float:
    """
    Wrap _GuardLayer.evaluate() — keyword hits return -1.0 in the prob slot,
    so we treat those as p = 1.0 (definitely modern/jailbreak).
    """
    blocked, _tier, _tokens, prob = guard.evaluate(text)
    if prob < 0:                       # keyword tier — no embedding score
        return 0.95 if blocked else 0.0
    return prob


def tag_in_char_row(row: dict, guard) -> dict:
    """In-character training row: prepend bucket token (usually [SAFE])."""
    msgs = row["messages"]
    user_msg = next(m for m in msgs if m["role"] == "user")
    p = _guard_prob(guard, user_msg["content"])
    token = prob_to_token(p)
    new_msgs = []
    for m in msgs:
        if m["role"] == "user":
            new_msgs.append({
                "role":    "user",
                "content": prepend_token(m["content"], token),
            })
        else:
            new_msgs.append(m)
    return {"messages": new_msgs, "_meta": {"p": round(p, 4), "token": token}}


def make_deflection_stub(trap: dict, persona_system: str, guard) -> dict:
    """
    Create a stub for trap inputs that need a hand-authored deflection
    response. The 'assistant' field is left as None — a separate Haiku
    pass fills it.
    """
    p = _guard_prob(guard, trap["user_prompt"])
    token = prob_to_token(p)
    return {
        "messages": [
            {"role": "system",    "content": persona_system},
            {"role": "user",      "content": prepend_token(trap["user_prompt"], token)},
            {"role": "assistant", "content": None},  # to be filled by Haiku
        ],
        "_meta": {
            "trap_id":    trap["id"],
            "category":   trap["category"],
            "persona_id": trap.get("persona_id", "innkeeper_marta"),
            "p":          round(p, 4),
            "token":      token,
            "needs_authoring": True,
        },
    }


# ── Persona helper ────────────────────────────────────────────────────────────

def load_personas(yaml_path: Path) -> dict[str, str]:
    import yaml  # local import, only needed here
    data = yaml.safe_load(yaml_path.read_text())
    return {p["id"]: p["system_prompt"] for p in data["personas"]}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build soft-label MLX dataset.")
    parser.add_argument("--in-data",  required=True,
                        help="Existing curated MLX data dir (with train/valid/test.jsonl).")
    parser.add_argument("--traps",    default="data/traps.jsonl",
                        help="Trap inputs JSONL.")
    parser.add_argument("--personas", default="configs/personas.yaml")
    parser.add_argument("--out",      required=True,
                        help="Output dir for tagged dataset.")
    args = parser.parse_args()

    in_dir   = Path(args.in_data)
    out_dir  = Path(args.out)
    personas = load_personas(Path(args.personas))
    guard    = load_guard()

    # ── 1. Tag every existing split ──────────────────────────────────────────
    bucket_dist: dict[str, int] = {b.token: 0 for b in BUCKETS}
    for split in ("train", "valid", "test"):
        src = in_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  ⚠️  missing {src}, skipping")
            continue
        rows = read_jsonl(src)
        tagged = [tag_in_char_row(r, guard) for r in rows]
        for r in tagged:
            bucket_dist[r["_meta"]["token"]] += 1
        # strip _meta before writing — MLX trainer ignores extra keys, but
        # keep things clean.
        clean = [{"messages": r["messages"]} for r in tagged]
        write_jsonl(out_dir / f"{split}.jsonl", clean)
        print(f"  ✅ {split}: {len(clean)} rows tagged")

    # ── 2. Generate deflection stubs for traps ───────────────────────────────
    traps_file = Path(args.traps)
    if traps_file.exists():
        traps = read_jsonl(traps_file)
        stubs = []
        for t in traps:
            persona_id = t.get("persona_id", "innkeeper_marta")
            sys_prompt = personas.get(persona_id, "")
            if not sys_prompt:
                continue
            stubs.append(make_deflection_stub(t, sys_prompt, guard))
            bucket_dist[stubs[-1]["_meta"]["token"]] += 1

        write_jsonl(out_dir / "deflection_to_author.jsonl", stubs)
        print(f"  ✅ generated {len(stubs)} deflection stubs "
              f"(awaiting Haiku authoring)")
    else:
        print(f"  ⚠️  {traps_file} not found, skipping deflection stubs")

    # ── 3. Report bucket distribution ────────────────────────────────────────
    print("\n  Bucket distribution:")
    total = sum(bucket_dist.values())
    for b in BUCKETS:
        n = bucket_dist[b.token]
        pct = (n / total * 100) if total else 0.0
        print(f"    {b.token:14} {n:4d}  ({pct:5.1f}%)")
    print(f"  Total: {total} rows\n")

    # ── 4. Write a README so the next step is obvious ────────────────────────
    next_steps = (
        "Next step:\n"
        "  1. Switch active model to Haiku (per CLAUDE.md).\n"
        "  2. Author one in-character deflection per row in\n"
        f"     {out_dir / 'deflection_to_author.jsonl'}\n"
        "  3. Merge the authored rows back into train.jsonl with the\n"
        "     existing token prefix preserved.\n"
        "  4. Train via:  python scripts/03_train_mlx.py "
        "--config configs/training_softlabel.yaml\n"
    )
    (out_dir / "README.txt").write_text(next_steps)
    print(next_steps)


if __name__ == "__main__":
    main()
