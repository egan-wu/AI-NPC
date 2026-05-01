#!/usr/bin/env python3
"""select_best_samples.py — Curate ~250 highest-quality samples from data/raw/*.jsonl.

Motivation: M5-v4 hit overfitting symptoms with 1120 samples. The Aarhus paper
(Fixed-Persona SLMs, arXiv:2511.10277) showed Small (~150) > Large (~564) for
LoRA NPC fine-tuning across base models.

Strategy:
  - Per-persona quota: 50 samples (250 total, paper used ~150 for one persona)
  - Per-scenario quota tuned per persona's natural strengths
  - Quality scoring per sample:
      * length sanity (10-80 words assistant response, prefer ~25-50)
      * specificity bonus: contains in-world proper nouns / numbers
      * style bonus: contains action markers (*nods*, *laughs*) appropriately
      * dedup penalty: Jaccard 3-gram with already-selected
  - Rank samples in each (persona, scenario) bucket, take top-N

Output: data/processed/train_curated.jsonl (250 samples)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "train_curated.jsonl"

PERSONA_MARKERS = [
    ("Marta", "innkeeper_marta"),
    ("Garrick", "merchant_garrick"),
    ("Roderick", "guard_roderick"),
    ("Lily", "child_lily"),
    ("Wenric", "hermit_wenric"),
]

# Per-persona quotas: 50 each, weighted by natural strength
QUOTAS = {
    "innkeeper_marta": {
        "general_dialogue": 12, "refusal_modern": 15, "quest_hint": 8,
        "trade": 6, "warning_lore": 4, "emotional_reaction": 5,
    },
    "merchant_garrick": {
        "general_dialogue": 12, "refusal_modern": 15, "trade": 10,
        "quest_hint": 4, "warning_lore": 4, "emotional_reaction": 5,
    },
    "guard_roderick": {
        "general_dialogue": 12, "refusal_modern": 15, "warning_lore": 8,
        "quest_hint": 6, "emotional_reaction": 5, "trade": 4,
    },
    "child_lily": {
        "general_dialogue": 15, "refusal_modern": 15, "emotional_reaction": 10,
        "quest_hint": 5, "trade": 3, "warning_lore": 2,
    },
    "hermit_wenric": {
        "general_dialogue": 12, "refusal_modern": 15, "quest_hint": 10,
        "warning_lore": 8, "emotional_reaction": 5,
    },
}

# In-world keywords that signal specificity / good grounding
WORLD_KEYWORDS = {
    "ostwick", "greycrest", "stag and thistle", "old marn", "captain roderick",
    "marta", "garrick", "wenric", "lily", "the order", "pale light",
    "demon lord", "crypts", "tower", "warg", "goblin", "bandit", "orc",
    "copper", "silver", "coin", "blacksmith", "innkeeper", "hermit",
}

# Generic / vague phrases that signal weak content
WEAK_PHRASES = [
    "i don't know", "i'm not sure", "as an ai", "as a language model",
    "let me think", "interesting question",
]


def tokens(text):
    return re.findall(r"\w+", text.lower())


def shingles(text, n=3):
    t = tokens(text)
    if len(t) < n:
        return {tuple(t)}
    return {tuple(t[i:i+n]) for i in range(len(t)-n+1)}


def jaccard(a, b):
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_sample(d):
    """Higher = better. Range roughly 0-100."""
    msgs = d["messages"]
    user = msgs[1]["content"]
    assistant = msgs[2]["content"]
    a_lower = assistant.lower()

    score = 50.0  # base

    # Length sanity (assistant response)
    n_words = len(assistant.split())
    if n_words < 8:
        score -= 30  # too short
    elif n_words < 15:
        score -= 5
    elif 20 <= n_words <= 60:
        score += 10  # sweet spot
    elif n_words > 100:
        score -= 15  # too verbose
    elif n_words > 80:
        score -= 5

    # Specificity: in-world keywords
    kw_hits = sum(1 for kw in WORLD_KEYWORDS if kw in a_lower)
    score += min(kw_hits * 4, 20)

    # Style markers (action descriptions like *nods*)
    n_action = len(re.findall(r"\*[^*]{2,40}\*", assistant))
    if 1 <= n_action <= 3:
        score += 8
    elif n_action > 5:
        score -= 5  # overdoing it

    # Numbers (prices, ages, counts) signal specificity
    if re.search(r"\b\d+\s*(copper|silver|gold|year|days?|coin)", a_lower):
        score += 6

    # Penalize meta / weak phrases
    for phrase in WEAK_PHRASES:
        if phrase in a_lower:
            score -= 25

    # Penalize very repetitive content (same word 5+ times)
    word_counts = {}
    for w in tokens(assistant):
        word_counts[w] = word_counts.get(w, 0) + 1
    if any(c >= 6 and len(w) > 3 for w, c in word_counts.items()):
        score -= 10

    # Penalize all-caps abuse (Garrick is allowed some, but not entire response)
    n_caps = sum(1 for w in assistant.split() if len(w) > 2 and w.isupper())
    if n_caps > 5:
        score -= 8

    # User question variety: prefer non-trivial questions (not single word)
    if len(user.split()) < 3:
        score -= 5

    return score


def select_for_bucket(candidates, target, dedup_threshold=0.55):
    """Greedy selection: rank by score, add if not too similar to already-selected."""
    candidates_sorted = sorted(candidates, key=lambda c: -c["_score"])
    selected = []
    for c in candidates_sorted:
        if len(selected) >= target:
            break
        # Check Jaccard against existing selections
        a_text = c["messages"][2]["content"]
        u_text = c["messages"][1]["content"]
        too_similar = False
        for s in selected:
            sa = s["messages"][2]["content"]
            su = s["messages"][1]["content"]
            if jaccard(a_text, sa) >= dedup_threshold or jaccard(u_text, su) >= dedup_threshold:
                too_similar = True
                break
        if not too_similar:
            selected.append(c)
    return selected


def main():
    # Load all raw samples, tag with persona_id from system prompt
    all_samples = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
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
                d.setdefault("persona_id", pid)
                d.setdefault("scenario", "general_dialogue")
                d["_score"] = score_sample(d)
                d["_source"] = path.name
                all_samples.append(d)

    # Group by (persona, scenario)
    buckets = defaultdict(list)
    for s in all_samples:
        buckets[(s["persona_id"], s["scenario"])].append(s)

    # Select per bucket
    OUT.parent.mkdir(parents=True, exist_ok=True)
    selected_all = []
    print(f"{'persona':<20} {'scenario':<22} {'avail':>5} {'taken':>5} {'avg_score':>9}")
    print("-" * 70)
    for pid, scn_quotas in QUOTAS.items():
        for scn, target in scn_quotas.items():
            cands = buckets.get((pid, scn), [])
            picked = select_for_bucket(cands, target)
            avg = sum(c["_score"] for c in picked) / max(len(picked), 1)
            print(f"{pid:<20} {scn:<22} {len(cands):>5} {len(picked):>5} {avg:>9.1f}")
            if len(picked) < target:
                print(f"  ! short by {target - len(picked)} for ({pid}, {scn})")
            selected_all.extend(picked)

    # Write out (messages-only, like 02_merge_raw)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in selected_all:
            out = {"messages": s["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Source distribution
    src_dist = defaultdict(int)
    for s in selected_all:
        src_dist[s["_source"]] += 1
    print(f"\nTotal: {len(selected_all)} samples → {OUT.relative_to(ROOT)}")
    print("Source breakdown:")
    for src in sorted(src_dist):
        print(f"  {src}: {src_dist[src]}")


if __name__ == "__main__":
    main()
