#!/usr/bin/env python3
"""prep_mistral_dataset.py — Convert mlx_curated_medium to Mistral-compatible format.

Mistral-7B-Instruct's chat template only supports user/assistant roles.
Fix: merge the system prompt into the first user message so the template
produces: [INST] {system}\n\n{user} [/INST] {assistant}

Reads:  data/mlx_curated_medium/{train,valid,test}.jsonl
Writes: data/mlx_mistral/{train,valid,test}.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "mlx_curated_medium"
DST = ROOT / "data" / "mlx_mistral"


def convert(sample):
    msgs = sample["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user_turns = [m for m in msgs if m["role"] in ("user", "assistant")]
    if system and user_turns and user_turns[0]["role"] == "user":
        # Prepend system prompt to first user turn
        merged = f"{system}\n\n{user_turns[0]['content']}"
        new_msgs = [{"role": "user", "content": merged}] + user_turns[1:]
    else:
        new_msgs = user_turns
    return {"messages": new_msgs}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        src = SRC / f"{split}.jsonl"
        dst = DST / f"{split}.jsonl"
        n = 0
        with open(src) as f_in, open(dst, "w") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                out = convert(sample)
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}.jsonl: {n} samples")
    print(f"\nWrote → {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
