#!/usr/bin/env python3
"""
01_generate_data.py — Expand 10 hand-written seeds into ~500 training samples
via Claude Haiku 4.5.

Reads:  configs/personas.yaml, data/seeds.jsonl
Writes: data/raw/<persona_id>.jsonl

Pipeline per (persona, scenario) cell:
  1. Build a cached system block: instructions + all 10 anchor seeds.
     The seeds + instructions stay constant across the run, so prompt
     caching makes input cost negligible after the first call.
  2. Ask Haiku for `target * 1.5` diverse user prompts in one call.
  3. Dedup user prompts via n-gram Jaccard (zero deps), keep top `target`.
  4. For each kept prompt: generate one assistant response.
     Strict mode — any failure (API error, empty, too short) → drop sample.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml
from anthropic import Anthropic
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
PERSONAS_PATH = ROOT / "configs" / "personas.yaml"
SEEDS_PATH = ROOT / "data" / "seeds.jsonl"
RAW_DIR = ROOT / "data" / "raw"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

# Per-(persona, scenario) sample quotas. Per-scenario sums match
# PROJECT_PLAN §4.3 (200/100/80/60/40/20). Distribution favors each persona's
# natural fit per personas.yaml — e.g. Garrick gets most `trade`, Wenric most
# `quest_hint`. Total = 500.
QUOTAS: Dict[str, Dict[str, int]] = {
    "innkeeper_marta": {
        "general_dialogue": 40, "quest_hint": 40, "trade": 15,
        "warning_lore": 5,  "refusal_modern": 8, "emotional_reaction": 2,
    },
    "merchant_garrick": {
        "general_dialogue": 40, "quest_hint": 5,  "trade": 50,
        "warning_lore": 3,  "refusal_modern": 8, "emotional_reaction": 0,
    },
    "guard_roderick": {
        "general_dialogue": 40, "quest_hint": 10, "trade": 5,
        "warning_lore": 25, "refusal_modern": 8, "emotional_reaction": 0,
    },
    "child_lily": {
        "general_dialogue": 40, "quest_hint": 5,  "trade": 5,
        "warning_lore": 2,  "refusal_modern": 8, "emotional_reaction": 10,
    },
    "hermit_wenric": {
        "general_dialogue": 40, "quest_hint": 40, "trade": 5,
        "warning_lore": 25, "refusal_modern": 8, "emotional_reaction": 8,
    },
}

# Two user prompts are "near duplicates" if their 3-gram Jaccard >= this.
# 0.6 is conservative — drops obvious paraphrases, keeps semantic variety.
JACCARD_DEDUP_THRESHOLD = 0.6

MIN_RESPONSE_WORDS = 4  # below this, treat as malformed and drop


# ---------- I/O helpers ----------

def load_seeds() -> List[dict]:
    with open(SEEDS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_personas() -> Dict[str, dict]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {p["id"]: p for p in cfg["personas"]}


# ---------- Dedup ----------

def shingles(text: str, n: int = 3) -> set:
    tokens = text.lower().split()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str) -> float:
    s1, s2 = shingles(a), shingles(b)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def dedup_prompts(prompts: List[str]) -> List[str]:
    """Keep prompts in order; drop any that's too similar to one already kept."""
    kept: List[str] = []
    for p in prompts:
        if all(jaccard(p, k) < JACCARD_DEDUP_THRESHOLD for k in kept):
            kept.append(p)
    return kept


# ---------- API calls ----------

def build_system_blocks(seeds: List[dict], task: str) -> List[dict]:
    """Build the system blocks list. The instructions + 10 seeds go in a
    cached block (constant across the run); the per-call task instruction
    is appended uncached."""
    seed_text = "\n\n---\n\n".join(
        f"persona_id: {s['persona_id']}\n"
        f"scenario: {s['scenario']}\n"
        f"system: {s['messages'][0]['content']}\n"
        f"user: {s['messages'][1]['content']}\n"
        f"assistant: {s['messages'][2]['content']}"
        for s in seeds
    )
    cached = (
        "You are a data generation assistant for an RPG NPC fine-tuning project.\n"
        "Below are 10 hand-written anchor examples that define the target voice "
        "and style for each NPC. Match the voice exactly — vocabulary, sentence "
        "rhythm, character quirks. Be specific (concrete names, places) like the "
        "anchors. Never break character. Never mention being an AI.\n\n"
        "=== ANCHOR EXAMPLES ===\n\n" + seed_text
    )
    return [
        {"type": "text", "text": cached, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": task},
    ]


def generate_user_prompts(
    client: Anthropic, seeds: List[dict], persona: dict, scenario: str, n: int
) -> List[str]:
    task = (
        f"Generate {n} diverse user (player) prompts that a fantasy hero might "
        f"say to {persona['id']} (a {persona['role']}) in a `{scenario}` scenario. "
        "Each prompt should explore a different angle — different topics, tones, "
        "lengths. Output ONLY a JSON array of strings, no other text. "
        'Example format: ["Where is the inn?", "Have you seen the priest?"]'
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=build_system_blocks(seeds, task),
        messages=[{"role": "user", "content": "Generate the prompts now."}],
    )
    text = resp.content[0].text.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    prompts = json.loads(text[start:end + 1])
    return [p for p in prompts if isinstance(p, str) and p.strip()]


def generate_response(
    client: Anthropic, seeds: List[dict], persona: dict, scenario: str, user_prompt: str
) -> str:
    task = (
        f"You will roleplay as {persona['id']}. The scenario is `{scenario}`. "
        "Below is the persona's system prompt — adopt this voice exactly. "
        "Then respond to the player's message in character. Output ONLY the "
        "in-character reply: no JSON, no narration, no surrounding quotes.\n\n"
        f"=== PERSONA SYSTEM PROMPT ===\n{persona['system_prompt']}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_blocks(seeds, task),
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


# ---------- Orchestration ----------

def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

    seeds = load_seeds()
    personas = load_personas()
    client = Anthropic()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_dropped = 0

    for persona_id, scenario_quotas in QUOTAS.items():
        persona = personas[persona_id]
        out_path = RAW_DIR / f"{persona_id}.jsonl"
        # Truncate on each run — safer than appending and getting silent dupes.
        # If you want to resume, comment this out and add a skip-if-exists check.
        with open(out_path, "w", encoding="utf-8") as out:
            for scenario, target in scenario_quotas.items():
                if target == 0:
                    continue
                print(f"[{persona_id}] {scenario}: target={target}", flush=True)

                # Step 1: get diverse user prompts (1.5x cushion for dedup loss).
                requested = max(int(target * 1.5), target + 5)
                try:
                    raw_prompts = generate_user_prompts(
                        client, seeds, persona, scenario, requested
                    )
                except Exception as e:
                    print(f"  ! batch user-prompt generation failed: {e}")
                    total_dropped += target
                    continue
                deduped = dedup_prompts(raw_prompts)[:target]
                if len(deduped) < target:
                    print(f"  ! only {len(deduped)} unique prompts after dedup")

                # Step 2: one assistant response per prompt. Strict drop.
                for user_prompt in deduped:
                    try:
                        assistant = generate_response(
                            client, seeds, persona, scenario, user_prompt
                        )
                    except Exception as e:
                        print(f"  ! response failed: {e}")
                        total_dropped += 1
                        continue
                    if not assistant or len(assistant.split()) < MIN_RESPONSE_WORDS:
                        total_dropped += 1
                        continue
                    sample = {
                        "messages": [
                            {"role": "system", "content": persona["system_prompt"]},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant},
                        ],
                        "persona_id": persona_id,
                        "scenario": scenario,
                        "tags": [],
                    }
                    out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    out.flush()  # so a crash mid-run still leaves partial output usable
                    total_kept += 1
                time.sleep(0.1)  # polite pacing

        print(f"  → wrote {out_path.relative_to(ROOT)}")

    print(f"\nDone. Kept: {total_kept}. Dropped: {total_dropped}.")


if __name__ == "__main__":
    main()
