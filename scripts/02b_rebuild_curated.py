#!/usr/bin/env python3
"""05_rebuild_curated.py — Rebuild data/processed/train_curated.jsonl.

Why this exists
---------------
The previous train_curated.jsonl (247 samples) was hand-picked before the v9
batches were generated, so 55 high-quality v9 samples never made it in. It
also kept some duplicate user/assistant pairs across v1↔v2 versions.

This script rebuilds the curated set from all data/raw/*.jsonl using a
deterministic ranking + dedup pass — no API calls, no hand-curation.

Selection logic per persona
---------------------------
1. Pool every (user, assistant) pair from all raw files for that persona.
2. Rank each pair by (version_priority, length_score):
     version_priority: v9 > v3 > v2 > v1   (newer batches were tuned better)
     length_score:     prefer 12–60 word replies; penalise <8 words except
                       for personas where terseness is in-character (Roderick).
3. Dedup:
     a. drop pairs whose assistant text already appeared (keep highest-rank).
     b. drop pairs whose user prompt already appeared (keep highest-rank).
4. Cap at TARGET_PER_PERSONA samples per persona.

Inputs:  data/raw/*.jsonl
Output:  data/processed/train_curated.jsonl
         data/processed/train_curated.report.txt   (provenance summary)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "train_curated.jsonl"
REPORT_PATH = ROOT / "data" / "processed" / "train_curated.report.txt"

# Persona detection from system prompt
PERSONA_MARKERS = [
    ("Marta",    "innkeeper_marta"),
    ("Garrick",  "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily",     "child_lily"),
    ("Wenric",   "hermit_wenric"),
]
PID_FROM_NAME = {name: pid for name, pid in PERSONA_MARKERS}

# Personas where short replies are in-character — don't penalise terseness
TERSE_PERSONAS = {"guard_roderick"}

# Target curated size per persona (47–50 in the previous curated set)
TARGET_PER_PERSONA = 60

# Version-batch priority: newer/better-tuned batches outrank older ones
VERSION_PRIORITY = {"v9": 4, "v3": 3, "v2": 2, "v1": 1}

_VERSION_RE = re.compile(r"_v(\d+)\.jsonl$")


def detect_version(filename: str) -> str:
    m = _VERSION_RE.search(filename)
    if not m:
        return "v1"  # base file (no _vN suffix) is the original v1 batch
    n = m.group(1)
    return f"v{n}" if f"v{n}" in VERSION_PRIORITY else "v1"


def detect_persona(system_prompt: str) -> str | None:
    for marker, pid in PERSONA_MARKERS:
        if marker in system_prompt:
            return pid
    return None


def length_score(assistant: str, persona_id: str) -> float:
    """Reward 12–60 word replies; penalise extremes — except terseness for Roderick."""
    n_words = len(assistant.split())
    if persona_id in TERSE_PERSONAS:
        # Roderick: 4–25 words is the sweet spot
        if 4 <= n_words <= 25:
            return 1.0
        if n_words < 4:
            return 0.3
        return 0.6  # over-talky guard is off-character
    # Other personas
    if 12 <= n_words <= 60:
        return 1.0
    if 8 <= n_words < 12:
        return 0.7
    if 60 < n_words <= 90:
        return 0.7
    return 0.3  # <8 or >90 words — likely too terse or rambling


def rank_key(version: str, persona_id: str, assistant: str) -> tuple[int, float]:
    """Higher tuple sorts first."""
    return (VERSION_PRIORITY.get(version, 0), length_score(assistant, persona_id))


def main() -> None:
    # Pool: persona_id -> list of candidate dicts
    pool: dict[str, list[dict]] = defaultdict(list)
    file_contributions: dict[str, int] = Counter()

    for path in sorted(RAW_DIR.glob("*.jsonl")):
        version = detect_version(path.name)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d["messages"]
                sys_text = next((m["content"] for m in msgs if m["role"] == "system"), "")
                user     = next((m["content"] for m in msgs if m["role"] == "user"), "")
                assist   = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                pid = detect_persona(sys_text)
                if not pid or not user or not assist:
                    continue
                pool[pid].append({
                    "messages": msgs,
                    "user": user,
                    "assistant": assist,
                    "version": version,
                    "source_file": path.name,
                    "rank": rank_key(version, pid, assist),
                })

    # Per-persona: sort by rank desc, dedup by assistant then user, cap.
    selected: list[dict] = []
    persona_kept: dict[str, int] = {}
    persona_sources: dict[str, Counter] = {pid: Counter() for pid in PID_FROM_NAME.values()}

    for pid in PID_FROM_NAME.values():
        cands = sorted(pool.get(pid, []), key=lambda c: c["rank"], reverse=True)
        seen_assist: set[str] = set()
        seen_user:   set[str] = set()
        kept: list[dict] = []
        for c in cands:
            a_norm = c["assistant"].strip()
            u_norm = c["user"].strip()
            if a_norm in seen_assist or u_norm in seen_user:
                continue
            seen_assist.add(a_norm)
            seen_user.add(u_norm)
            kept.append(c)
            if len(kept) >= TARGET_PER_PERSONA:
                break
        persona_kept[pid] = len(kept)
        for c in kept:
            persona_sources[pid][c["source_file"]] += 1
            file_contributions[c["source_file"]] += 1
        selected.extend(kept)

    # Write curated jsonl (messages-only, matching prior format)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for c in selected:
            fh.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

    # Provenance report
    lines: list[str] = []
    lines.append(f"Rebuilt {OUT_PATH.relative_to(ROOT)}")
    lines.append(f"Total samples: {len(selected)}")
    lines.append("")
    lines.append("Per-persona kept:")
    for pid, n in persona_kept.items():
        lines.append(f"  {pid:20s} {n}")
    lines.append("")
    lines.append("Per-persona source breakdown (source_file: kept_count):")
    for pid, ctr in persona_sources.items():
        lines.append(f"  [{pid}]")
        for f, n in sorted(ctr.items()):
            lines.append(f"    {f:40s} {n}")
    lines.append("")
    lines.append("File contribution totals:")
    for f, n in sorted(file_contributions.items()):
        lines.append(f"  {f:40s} {n}")
    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\n→ wrote {len(selected)} samples to {OUT_PATH.relative_to(ROOT)}")
    print(f"→ wrote provenance to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
