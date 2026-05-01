#!/usr/bin/env python3
"""prep_medium_prompts.py — Rewrite curated MLX data with medium-length system
prompts (configs/personas_medium.yaml).

Reads:  data/mlx_curated/{train,valid,test}.jsonl
Writes: data/mlx_curated_medium/{train,valid,test}.jsonl
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "mlx_curated"
DST = ROOT / "data" / "mlx_curated_medium"
PERSONAS_MED = ROOT / "configs" / "personas_medium.yaml"

PERSONA_MARKERS = [
    ("Marta", "innkeeper_marta"),
    ("Garrick", "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily", "child_lily"),
    ("Wenric", "hermit_wenric"),
]


def main():
    with open(PERSONAS_MED) as f:
        med = {p["id"]: p["system_prompt"].strip() for p in yaml.safe_load(f)["personas"]}

    DST.mkdir(parents=True, exist_ok=True)
    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        in_path = SRC / split
        out_path = DST / split
        n = 0
        with open(in_path) as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sys_text = d["messages"][0]["content"]
                pid = None
                for marker, p in PERSONA_MARKERS:
                    if marker in sys_text:
                        pid = p
                        break
                if pid is None:
                    continue
                d["messages"][0]["content"] = med[pid]
                f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n}")
    print(f"\nWrote → {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
