#!/usr/bin/env python3
"""prep_short_prompts.py — Rewrite curated MLX data with short system prompts.

For M5-v6: replaces the long ~800-token persona prompts (which the model was
memorizing and parroting back) with one-line identity prompts from
configs/personas_short.yaml.

Reads:  data/mlx_curated/{train,valid,test}.jsonl
Writes: data/mlx_curated_short/{train,valid,test}.jsonl
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "mlx_curated"
DST = ROOT / "data" / "mlx_curated_short"
PERSONAS_SHORT = ROOT / "configs" / "personas_short.yaml"

PERSONA_MARKERS = [
    ("Marta", "innkeeper_marta"),
    ("Garrick", "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily", "child_lily"),
    ("Wenric", "hermit_wenric"),
]


def main():
    with open(PERSONAS_SHORT) as f:
        short = {p["id"]: p["system_prompt"] for p in yaml.safe_load(f)["personas"]}

    DST.mkdir(parents=True, exist_ok=True)

    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        in_path = SRC / split
        out_path = DST / split
        n_in = n_out = 0
        with open(in_path) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                d = json.loads(line)
                sys_text = d["messages"][0]["content"]
                pid = None
                for marker, p in PERSONA_MARKERS:
                    if marker in sys_text:
                        pid = p
                        break
                if pid is None:
                    print(f"  WARN: cannot identify persona, skipping: {sys_text[:60]}")
                    continue
                d["messages"][0]["content"] = short[pid]
                f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_out += 1
        print(f"{split}: {n_in} → {n_out}")

    print(f"\nWrote → {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
