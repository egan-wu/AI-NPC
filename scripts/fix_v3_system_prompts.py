#!/usr/bin/env python3
"""fix_v3_system_prompts.py — Replace non-standard system prompts in all *_v3.jsonl files.

v3 samples were generated with simplified system prompts (missing Voice/Knowledge/
Critical refusal rule sections). This script patches them in-place to use the
canonical system_prompt from configs/personas.yaml, keeping all other fields intact.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PERSONAS_PATH = ROOT / "configs" / "personas.yaml"

PERSONA_MARKERS = [
    ("Marta", "innkeeper_marta"),
    ("Garrick", "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily", "child_lily"),
    ("Wenric", "hermit_wenric"),
]


def main():
    with open(PERSONAS_PATH) as f:
        personas = {p["id"]: p for p in yaml.safe_load(f)["personas"]}

    v3_files = sorted(RAW_DIR.glob("*_v3.jsonl"))
    print(f"Found {len(v3_files)} v3 files to patch\n")

    total_fixed = 0
    total_skipped = 0

    for path in v3_files:
        lines_in = path.read_text(encoding="utf-8").splitlines()
        lines_out = []
        fixed = skipped = 0

        for line in lines_in:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sys_text = d["messages"][0]["content"]

            persona_id = None
            for marker, pid in PERSONA_MARKERS:
                if marker in sys_text:
                    persona_id = pid
                    break

            if persona_id is None:
                print(f"  WARNING: could not identify persona in {path.name}: {sys_text[:60]}")
                lines_out.append(json.dumps(d, ensure_ascii=False))
                skipped += 1
                continue

            canonical = personas[persona_id]["system_prompt"].strip()
            if sys_text.strip() == canonical:
                lines_out.append(json.dumps(d, ensure_ascii=False))
                # already correct
            else:
                d["messages"][0]["content"] = canonical
                lines_out.append(json.dumps(d, ensure_ascii=False))
                fixed += 1

        # write back
        path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        print(f"  {path.name}: {fixed} fixed, {skipped} skipped")
        total_fixed += fixed
        total_skipped += skipped

    print(f"\nDone. Total fixed: {total_fixed}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
