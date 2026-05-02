"""
14b_merge_deflect.py — Merge authored deflection responses into train/valid/test splits.

Run this AFTER a Haiku pass has filled assistant=None in
data/lora_deflect/deflection_to_author.jsonl.

Usage:
    python scripts/14b_merge_deflect.py \
        --authored  data/lora_deflect/deflection_to_author.jsonl \
        --out       data/lora_deflect \
        --seed      42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authored",  default="data/lora_deflect/deflection_to_author.jsonl")
    parser.add_argument("--out",       default="data/lora_deflect")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--valid-pct", type=float, default=0.10)
    args = parser.parse_args()

    rows_raw = read_jsonl(Path(args.authored))

    # Validate — reject any row still missing assistant content
    complete, missing = [], []
    for r in rows_raw:
        asst = next((m for m in r["messages"] if m["role"] == "assistant"), None)
        if asst and asst["content"]:
            # Strip _meta before writing to training split
            clean = {"messages": [
                m for m in r["messages"]
                if m.get("content") is not None
            ]}
            complete.append(clean)
        else:
            missing.append(r.get("_meta", {}).get("trap_id", "?"))

    if missing:
        print(f"  ⚠️  {len(missing)} rows still have assistant=None: {missing}")
        print("  Fix them before merging.")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(complete)
    n = len(complete)
    n_train = int(n * args.train_pct)
    n_valid = int(n * args.valid_pct)

    splits = {
        "train": complete[:n_train],
        "valid": complete[n_train:n_train + n_valid],
        "test":  complete[n_train + n_valid:],
    }

    out_dir = Path(args.out)
    for split, rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        print(f"  ✅ {split}: {len(rows)} rows")

    print(f"\n  Total: {n} rows → train/valid/test in {out_dir}/")


if __name__ == "__main__":
    main()
