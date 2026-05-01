#!/usr/bin/env python3
"""02_merge_raw.py — Merge all data/raw/*.jsonl into data/processed/train.jsonl.

Strips persona_id / scenario / tags metadata so processed files stay
messages-only (consistent with what prep_mlx_data.py and MLX-LM expect).

Run this whenever new raw files are added before running prep_mlx_data.py.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "train.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    persona_counts: Counter = Counter()
    total = 0

    PERSONA_MARKERS = [
        ("Marta", "innkeeper_marta"),
        ("Garrick", "merchant_garrick"),
        ("Roderick", "guard_roderick"),
        ("Lily", "child_lily"),
        ("Wenric", "hermit_wenric"),
    ]

    with open(OUT, "w", encoding="utf-8") as out_f:
        for raw_file in sorted(RAW_DIR.glob("*.jsonl")):
            with open(raw_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    sys_text = d["messages"][0]["content"]
                    for marker, pid in PERSONA_MARKERS:
                        if marker in sys_text:
                            persona_counts[pid] += 1
                            break
                    # Keep messages only; strip generation metadata
                    out_f.write(json.dumps({"messages": d["messages"]}, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Wrote {total} samples → {OUT.relative_to(ROOT)}")
    print("Persona breakdown:")
    for pid, count in sorted(persona_counts.items()):
        print(f"  {pid}: {count}")


if __name__ == "__main__":
    main()
