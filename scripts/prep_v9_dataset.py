#!/usr/bin/env python3
"""prep_v9_dataset.py — Build data/mlx_curated_v9 for M5-v9 training.

Strategy:
  - Take all 247 samples from data/mlx_curated_medium/train.jsonl (medium prompts)
  - Convert the 55 new v9 raw samples to medium prompts (same as prep_medium_prompts.py)
  - Put 44 of 55 into train, 6 into valid, 5 into test (proportional to existing split)
  - valid/test also augmented with a few v9 samples so eval distribution matches train

Final counts (approx):
  train: 247 + 44 = 291
  valid:  25 +  6 =  31
  test:   25 +  5 =  30
"""
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC_MEDIUM = ROOT / "data" / "mlx_curated_medium"
RAW = ROOT / "data" / "raw"
DST = ROOT / "data" / "mlx_curated_v9"
PERSONAS_MED = ROOT / "configs" / "personas_medium.yaml"

PERSONA_MARKERS = [
    ("Marta", "innkeeper_marta"),
    ("Garrick", "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily", "child_lily"),
    ("Wenric", "hermit_wenric"),
]

V9_FILES = [
    RAW / "innkeeper_marta_v9.jsonl",
    RAW / "merchant_garrick_v9.jsonl",
    RAW / "guard_roderick_v9.jsonl",
    RAW / "child_lily_v9.jsonl",
    RAW / "hermit_wenric_v9.jsonl",
]

random.seed(42)


def load_medium_prompts():
    with open(PERSONAS_MED) as f:
        return {p["id"]: p["system_prompt"].strip() for p in yaml.safe_load(f)["personas"]}


def get_persona_id(sys_text):
    for marker, pid in PERSONA_MARKERS:
        if marker in sys_text:
            return pid
    return None


def convert_to_medium(sample, med_prompts):
    """Replace system prompt with medium version."""
    d = json.loads(json.dumps(sample))  # deep copy
    sys_text = d["messages"][0]["content"]
    pid = get_persona_id(sys_text)
    if pid is None:
        return None
    d["messages"][0]["content"] = med_prompts[pid]
    return d


def main():
    med = load_medium_prompts()

    # Load existing mlx_curated_medium splits
    existing = {}
    for split in ("train", "valid", "test"):
        path = SRC_MEDIUM / f"{split}.jsonl"
        existing[split] = [json.loads(l) for l in open(path) if l.strip()]

    # Load all 55 new v9 samples and convert to medium prompts
    v9_samples = []
    for fpath in V9_FILES:
        for line in open(fpath):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            converted = convert_to_medium(raw, med)
            if converted:
                v9_samples.append(converted)

    print(f"v9 new samples: {len(v9_samples)}")

    # Shuffle and split v9 samples: 44 train / 6 valid / 5 test
    random.shuffle(v9_samples)
    v9_train = v9_samples[:44]
    v9_valid = v9_samples[44:50]
    v9_test  = v9_samples[50:]

    splits = {
        "train": existing["train"] + v9_train,
        "valid": existing["valid"] + v9_valid,
        "test":  existing["test"]  + v9_test,
    }

    # Shuffle train
    random.shuffle(splits["train"])

    DST.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        out_path = DST / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{split}.jsonl: {len(samples)} samples → {out_path.relative_to(ROOT)}")

    print(f"\nDone. Total train: {len(splits['train'])}")


if __name__ == "__main__":
    main()
