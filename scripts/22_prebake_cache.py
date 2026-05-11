"""
22_prebake_cache.py — Offline KV Cache Baking Script  (方案 γ, PHASE_5_PLAN.md)

Pre-computes the KV cache for a persona's full static prefix:
    system_prompt  +  ALL world_knowledge facts  +  opening cue
and serialises the result to outputs/prebaked_cache/ as .safetensors.

On next startup, 20_npc_cli_memory.py loads this file instead of recomputing,
reducing cold-start TTFT from O(tokens / prefill_speed) to O(file_size / IO).

When to re-bake
---------------
- After editing system_prompt in personas.yaml
- After adding / updating world_knowledge entries
- After switching to a different base model

Usage
-----
    source .venv/bin/activate
    python scripts/22_prebake_cache.py -p marta           # bake Marta
    python scripts/22_prebake_cache.py -p marta --force   # re-bake even if cache exists
    python scripts/22_prebake_cache.py --all              # bake every persona
    python scripts/22_prebake_cache.py --all --force

Pre-baked prompt topology (Phase 6 hierarchy: L0 + L3)
------------------------------------------------------
    <s>[INST] {system_prompt}

    [World — common knowledge]      ← L0, shared (configs/world_knowledge.yaml)
    - fact_1
    - …

    [About me]                      ← L3, this NPC (personas.yaml)
    - fact_1
    - …

    {opening_cue} [/INST] {opening_response} </s>

L_p (player_lore) and L4 (conv) are dynamic and injected per-turn at runtime;
they are NOT baked.

The generated opening_response is stored in the .json sidecar so the runtime
can display it immediately without generating again.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache

from src.cache_utils import prebaked_path, save_cache, is_valid, adapter_id_from_path
from src.memory_hierarchy import HierarchicalMemory, WorldKnowledgeStore

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID      = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
PERSONAS_PATH = _root / "configs" / "personas.yaml"

PERSONA_DISPLAY = {
    "innkeeper_marta":  ("Marta",    "Stag & Thistle Inn"),
    "merchant_garrick": ("Garrick",  "Travelling Merchant"),
    "guard_roderick":   ("Roderick", "Captain of the Watch"),
    "child_lily":       ("Lily",     "Baker's Daughter"),
    "hermit_wenric":    ("Wenric",   "Sage Hermit of Greycrest"),
}

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"

OPENING_CUE = "(The traveller approaches. Give a short in-character greeting.)"


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_personas() -> dict:
    with open(PERSONAS_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {p["id"]: p for p in cfg["personas"]}


def resolve_persona(arg: str, personas: dict) -> str:
    if arg in personas:
        return arg
    for pid in personas:
        name, _ = PERSONA_DISPLAY[pid]
        if arg.lower() in pid.lower() or arg.lower() in name.lower():
            return pid
    print(f"{RED}Unknown persona '{arg}'. Available: {', '.join(personas)}{RESET}")
    sys.exit(1)


def format_block(label: str, facts: list[str]) -> str:
    """Format a list of facts as a labelled bullet block, or '' if empty."""
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts)
    return f"[{label}]\n{lines}"


def bake_persona(
    persona_id: str,
    persona_data: dict,
    model,
    tokenizer,
    force: bool,
    world_facts: list[str],
    adapter_id: str = "base",
    adapter_path: str | None = None,
) -> None:
    """
    Pre-bake the KV cache for one persona.

    Static prefix layout (Phase 6 hierarchy — bakes L0 + L3, NOT L_p / L4):
        [INST] {system_prompt}

        [World — common knowledge]   ← L0, shared across NPCs
        - …

        [About me]                   ← L3, this NPC's persona lore
        - …

        {opening_cue} [/INST] {opening_response} </s>

    L_p (player_lore) and L4 (conv) are dynamic and live in the per-turn β
    delta, not in this prefix.
    """
    npc_name, location = PERSONA_DISPLAY[persona_id]
    out_path = prebaked_path(persona_id, MODEL_ID, adapter_id=adapter_id)

    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  Baking: {BOLD}{CYAN}{npc_name}{RESET}  ·  {location}")
    print(f"  Adapter: {DIM}{adapter_id}{RESET}")
    print(f"  Output:  {DIM}{out_path}{RESET}")

    if not force and is_valid(out_path, persona_id, MODEL_ID, adapter_id=adapter_id):
        print(f"  {YELLOW}Cache already exists and is valid. Use --force to re-bake.{RESET}")
        return

    system_prompt   = persona_data["system_prompt"].strip()
    persona_lore    = persona_data.get("world_knowledge", [])  # L3 source
    world_block     = format_block("World — common knowledge", world_facts)
    persona_block   = format_block("About me", persona_lore)

    # Compose the full [INST] content for the static prefix
    static_content_parts = [system_prompt]
    if world_block:
        static_content_parts.append(world_block)
    if persona_block:
        static_content_parts.append(persona_block)
    static_content_parts.append(OPENING_CUE)
    static_content = "\n\n".join(static_content_parts)

    opening_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": static_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    prefix_token_ids = tokenizer.encode(opening_prompt, add_special_tokens=False)
    print(f"  Static prefix: {len(prefix_token_ids)} tokens "
          f"(world={len(world_facts)}, persona={len(persona_lore)} facts baked)")
    print(f"\n  {CYAN}{BOLD}[{npc_name}]{RESET} ", end="", flush=True)

    cache = make_prompt_cache(model)
    sampler   = make_sampler(temp=0.75)
    logit_prs = make_logits_processors(repetition_penalty=1.1)

    opening_parts: list[str] = []
    for chunk in stream_generate(
        model, tokenizer,
        prompt=opening_prompt,
        prompt_cache=cache,
        max_tokens=80,
        sampler=sampler,
        logits_processors=logit_prs,
    ):
        text = chunk.text if hasattr(chunk, "text") else chunk
        print(text, end="", flush=True)
        opening_parts.append(text)
    opening = "".join(opening_parts).strip()
    print("\n")

    metadata = {
        "npc_id":              persona_id,
        "model_id":            MODEL_ID,
        "adapter_id":          adapter_id,
        "adapter_path":        adapter_path or "",
        "world_facts_count":   len(world_facts),
        "persona_facts_count": len(persona_lore),
        "prefix_tokens":       len(prefix_token_ids),
        "baked_at":            date.today().isoformat(),
    }
    save_cache(cache, opening, out_path, metadata)
    print(f"  {GREEN}✓ Done.{RESET}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-bake KV cache for NPC static prefix (方案 γ)."
    )
    parser.add_argument("--persona", "-p", default=None,
                        help="Persona id or alias (e.g. 'marta'). Ignored if --all.")
    parser.add_argument("--all", action="store_true",
                        help="Bake all personas sequentially.")
    parser.add_argument("--force", action="store_true",
                        help="Re-bake even if a valid cache already exists.")
    parser.add_argument("--adapter", default=None,
                        help="Path to a LoRA adapter directory (e.g. "
                             "outputs/adapters/v9_curated_v9). "
                             "Adapter id (dir basename) is recorded in cache "
                             "metadata; different adapters need separate caches. "
                             "Omit to bake against the base model.")
    args = parser.parse_args()

    if not args.all and args.persona is None:
        parser.error("Provide --persona / -p or use --all.")

    personas = load_personas()

    if args.all:
        targets = list(personas.keys())
    else:
        targets = [resolve_persona(args.persona, personas)]

    # Load shared L0 facts ONCE so every persona bakes against the same lore
    world_store = WorldKnowledgeStore()
    world_facts = world_store.all_texts()
    if not world_facts:
        print(f"{YELLOW}Warning: world_global is empty. "
              f"Run scripts/23_seed_world_knowledge.py first if you want shared "
              f"world lore baked in.{RESET}")
    else:
        print(f"{DIM}Loaded {len(world_facts)} L0 world facts from world_global.{RESET}")

    adapter_id = adapter_id_from_path(args.adapter)
    if args.adapter:
        if not Path(args.adapter).exists():
            print(f"{RED}Adapter path not found: {args.adapter}{RESET}")
            sys.exit(1)
        print(f"{DIM}Loading {MODEL_ID} + adapter [{adapter_id}]...{RESET}", end=" ", flush=True)
        model, tokenizer = load(MODEL_ID, adapter_path=args.adapter)
    else:
        print(f"{YELLOW}No --adapter given; baking against base model.{RESET}")
        print(f"{DIM}Loading {MODEL_ID}...{RESET}", end=" ", flush=True)
        model, tokenizer = load(MODEL_ID)
    print("ready.")

    for pid in targets:
        bake_persona(pid, personas[pid], model, tokenizer,
                     force=args.force, world_facts=world_facts,
                     adapter_id=adapter_id, adapter_path=args.adapter)

    print(f"\n{BOLD}All done.{RESET}  Adapter id: {DIM}{adapter_id}{RESET}")
    print(f"  Run: python scripts/20_npc_cli_memory.py -p <name>"
          f"{' --adapter ' + args.adapter if args.adapter else ''}")


if __name__ == "__main__":
    main()
