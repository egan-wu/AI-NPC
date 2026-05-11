#!/usr/bin/env python3
"""prep_curated_mistral.py — Stratified split + Mistral chat-template prep.

Reads:  data/processed/train_curated.jsonl   (300 samples, 60 per persona)
Writes: data/mlx_curated_mistral/{train,valid}.jsonl

Two operations in one pass:
  1. Stratified split by persona — 90 % train (~54/persona) / 10 % valid (~6/persona).
     Each persona contributes proportionally to both splits so val loss
     reflects the same persona distribution as train.
  2. Mistral chat-template merge — Mistral-7B-Instruct has no system role.
     Merge `system_prompt` into the first user message so the chat template
     emits: [INST] {system}\n\n{user} [/INST] {assistant}

Determinism:
  - random.seed(42) before shuffling — same input always produces same split
  - Persona id is detected by scanning system_prompt for the canonical name
    (matches the rest of the pipeline; see 02_merge_raw.py)

Usage:
  source .venv/bin/activate
  python scripts/prep_curated_mistral.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SRC   = _ROOT / "data" / "processed" / "train_curated.jsonl"
DST   = _ROOT / "data" / "mlx_curated_mistral"

VALID_FRACTION = 0.10   # 10 % per persona → ~6 samples each → 30 total
SEED = 42

PERSONA_MARKERS = [
    ("Marta",    "innkeeper_marta"),
    ("Garrick",  "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily",     "child_lily"),
    ("Wenric",   "hermit_wenric"),
]


def detect_persona(system_text: str) -> str | None:
    for marker, pid in PERSONA_MARKERS:
        if marker in system_text:
            return pid
    return None


def merge_system_into_user(sample: dict) -> dict:
    """Mistral chat template fix: prepend system prompt to first user turn."""
    msgs = sample["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    other  = [m for m in msgs if m["role"] in ("user", "assistant")]
    if system and other and other[0]["role"] == "user":
        merged = f"{system}\n\n{other[0]['content']}"
        new_msgs = [{"role": "user", "content": merged}] + other[1:]
    else:
        new_msgs = other
    return {"messages": new_msgs}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    # Bucket by persona for stratified split
    buckets: dict[str, list[dict]] = defaultdict(list)
    for line in open(SRC, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        sys_txt = next((m["content"] for m in sample["messages"]
                        if m["role"] == "system"), "")
        pid = detect_persona(sys_txt)
        if not pid:
            continue
        buckets[pid].append(sample)

    rng = random.Random(SEED)
    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    for pid in sorted(buckets):
        samples = buckets[pid][:]
        rng.shuffle(samples)
        n_valid = max(1, round(len(samples) * VALID_FRACTION))
        valid_rows.extend(samples[:n_valid])
        train_rows.extend(samples[n_valid:])

    # Shuffle the concatenated splits so personas interleave in batches
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    def write(rows: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(merge_system_into_user(r),
                                    ensure_ascii=False) + "\n")

    write(train_rows, DST / "train.jsonl")
    write(valid_rows, DST / "valid.jsonl")

    print(f"Wrote {DST.relative_to(_ROOT)}/")
    print(f"  train.jsonl: {len(train_rows)} samples")
    print(f"  valid.jsonl: {len(valid_rows)} samples")
    print(f"  Per-persona breakdown:")
    for pid in sorted(buckets):
        n_total = len(buckets[pid])
        n_val   = max(1, round(n_total * VALID_FRACTION))
        print(f"    {pid:20s} train={n_total - n_val}  valid={n_val}")


if __name__ == "__main__":
    main()
