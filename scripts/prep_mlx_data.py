#!/usr/bin/env python3
"""Prepare MLX-LM data directories from data/processed/{train,eval}.jsonl.

MLX-LM expects a directory containing exactly `train.jsonl`, `valid.jsonl`,
and (optionally) `test.jsonl`. This script materializes two such dirs:

  data/mlx/        — full M4 training: train=450, valid=25, test=25
  data/mlx_pilot/  — M3 pilot: train=80 (stratified by persona), valid=20,
                     test=25 (same test set as mlx/, for fair perplexity)

Deterministic via SEED=42.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(p, items):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    train = load(ROOT / "data/processed/train.jsonl")
    eval_ = load(ROOT / "data/processed/eval.jsonl")

    rng = random.Random(SEED)

    # --- Full M4 dataset ---
    # Split eval into valid + test (25 each); ignore any extras
    rng.shuffle(eval_)
    valid_full = eval_[:25]
    test_full = eval_[25:50]
    save(ROOT / "data/mlx/train.jsonl", train)
    save(ROOT / "data/mlx/valid.jsonl", valid_full)
    save(ROOT / "data/mlx/test.jsonl", test_full)

    # --- Pilot M3 dataset ---
    # Take min(15, available) per persona for some balance, then shuffle.
    # Roderick is undersampled (only ~8 train) — take whatever we have.
    # Samples don't carry persona_id, so infer from the system prompt.
    PERSONA_MARKERS = [
        ("Marta", "innkeeper_marta"),
        ("Garrick", "merchant_garrick"),
        ("Roderick", "guard_roderick"),
        ("Lily", "child_lily"),
        ("Wenric", "hermit_wenric"),
    ]
    by_persona = defaultdict(list)
    for s in train:
        sys_text = s["messages"][0]["content"]
        for marker, pid in PERSONA_MARKERS:
            if marker in sys_text:
                by_persona[pid].append(s)
                break

    pilot = []
    for pid, samples in sorted(by_persona.items()):
        rng.shuffle(samples)
        pilot.extend(samples[:15])
    rng.shuffle(pilot)

    pilot_train = pilot[: int(len(pilot) * 0.8)]
    pilot_valid = pilot[int(len(pilot) * 0.8):]

    save(ROOT / "data/mlx_pilot/train.jsonl", pilot_train)
    save(ROOT / "data/mlx_pilot/valid.jsonl", pilot_valid)
    save(ROOT / "data/mlx_pilot/test.jsonl", test_full)  # share test for fair perplexity

    print(f"M4 full:  train={len(train)}, valid={len(valid_full)}, test={len(test_full)}")
    print(f"M3 pilot: train={len(pilot_train)}, valid={len(pilot_valid)}, test={len(test_full)} (shared)")
    print(f"  per-persona pilot draw: {{p: min(15, len(s)) for p,s in by_persona.items()}}")
    for pid, samples in sorted(by_persona.items()):
        print(f"    {pid}: {len(samples)} avail → took {min(15, len(samples))}")


if __name__ == "__main__":
    main()
