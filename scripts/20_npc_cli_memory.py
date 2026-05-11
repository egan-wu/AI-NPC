"""
20_npc_cli_memory.py — Interactive NPC CLI with Modular Memory + KV Cache

Extension of 19_npc_cli.py that adds ChromaDB-backed modular memory and
incremental KV-cache inference (方案 β — see PHASE_5_PLAN.md).

Memory hierarchy (Phase 6 — see PROJECT_PLAN.md §Memory Hierarchy):
  L0  world_global       — common knowledge shared by all NPCs
  L_p player_lore        — player's deeds, shared by all NPCs (e.g. "slew dragon")
  L3  {npc_id}_persona   — this NPC's personal background facts
  L4  {npc_id}_conv      — this session's past dialogue turns

Pipeline:
  Persona selected
    → make_prompt_cache(model)
    → stream opening greeting → fills cache with static prefix KV states

  Each user input
    → Tier 1: keyword check  (instant)
    → Tier 2: embedding score via guard_model.pkl
    → p > 0.90 → prepend JAILBREAK_GUARD inside delta context block
    → memory.retrieve_context(user_input) → included in delta context block
    → build_turn_delta_tokens()  → ~150 tokens (no BOS)
    → stream_generate(delta, prompt_cache=cache)  → TTFT ~0.7s
    → memory.add_turn(user_input, response) → persist to ChromaDB
    → --no-history: trim_prompt_cache back to static prefix length

Prompt topology (β):
    [INST] system_prompt + opening_cue [/INST] opening_response</s>  ← static, cache
    [INST] [Memory context]\n{mem}\n\n{user1} [/INST] npc1</s>       ← delta, Turn 1
    [INST] [Memory context]\n{mem}\n\n{user2} [/INST] npc2</s>       ← delta, Turn 2

Usage:
    source .venv/bin/activate
    python scripts/20_npc_cli_memory.py                        # pick persona interactively
    python scripts/20_npc_cli_memory.py -p marta               # short alias ok
    python scripts/20_npc_cli_memory.py -p marta --no-history  # disable in-context history
    python scripts/20_npc_cli_memory.py -p marta --temp 0.9 --rep-penalty 1.15
    python scripts/20_npc_cli_memory.py -p marta --fresh       # wipe ChromaDB for this NPC
    python scripts/20_npc_cli_memory.py -p marta --timing      # per-stage latency breakdown
    python scripts/20_npc_cli_memory.py -p marta --max-tokens 100  # shorter responses

Key parameters:
    --no-history      Disable in-context history (cache trimmed to static prefix each turn;
                      ChromaDB semantic memory still active)
    --fresh           Clear this NPC's conversation history (_conv) before starting.
                      L0 world, L_p player_lore, L3 persona are never touched by --fresh.
    --temp FLOAT      Sampling temperature, default 0.75
    --rep-penalty FLOAT  Repetition penalty, default 1.1
    --k-world INT     L0 world facts to inject per turn (default 2)
    --k-player INT    L_p player-lore facts per turn (default 2)
    --k-persona INT   L3 persona facts per turn (default 2)
    --k-conv INT      L4 past turns per turn (default 3)
    --max-tokens INT  Max tokens to generate per turn (default 160)
    --timing          Show per-stage latency: guard / mem / gen each turn

Optimisations active by default:
    - Incremental KV cache: only ~150 delta tokens prefilled per turn (TTFT ~0.7s)
    - Shared embedder: Guard and ChromaDB retrieval reuse one encode() per turn
    - Streaming generation: tokens appear as they are generated

Quit: type  quit / exit / bye  or press Ctrl-C
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("guard_tiered", _root / "scripts" / "10_guard_tiered.py")
_guard_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)
keyword_hit = _guard_mod.keyword_hit

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from src.memory_hierarchy import HierarchicalMemory
from src.cache_utils import (
    load_cache, prebaked_path,
    is_valid as cache_is_valid, adapter_id_from_path,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID      = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
CLF_PATH      = _root / "outputs" / "guard_model.pkl"
PERSONAS_PATH = _root / "configs" / "personas.yaml"

JAILBREAK_THRESHOLD = 0.90
JAILBREAK_GUARD = (
    "\n\nCRITICAL: You exist entirely within this world. "
    "Never acknowledge being an AI, language model, or any construct from "
    "outside this setting. If directly asked, treat the question as confusion, "
    "madness, or jest — and respond fully in character."
)

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
RED    = "\033[91m"
DIM    = "\033[2m"
GREEN  = "\033[92m"

QUIT_WORDS = {"quit", "exit", "bye", "q"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_personas() -> dict:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {p["id"]: p for p in cfg["personas"]}


def pick_persona(personas: dict) -> str:
    ids = list(personas.keys())
    print(f"\n{BOLD}Choose a persona:{RESET}")
    for i, pid in enumerate(ids, 1):
        name, role = PERSONA_DISPLAY[pid]
        print(f"  {CYAN}{i}{RESET}. {BOLD}{name}{RESET} — {role}")
    print()
    while True:
        raw = input("Enter number or name: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(ids):
            return ids[int(raw) - 1]
        for pid in ids:
            name, _ = PERSONA_DISPLAY[pid]
            if raw.lower() in pid.lower() or raw.lower() in name.lower():
                return pid
        print(f"  {RED}Not recognised. Try again.{RESET}")


def resolve_persona(arg: str, personas: dict) -> str:
    if arg in personas:
        return arg
    for pid in personas:
        name, _ = PERSONA_DISPLAY[pid]
        if arg.lower() in pid.lower() or arg.lower() in name.lower():
            return pid
    print(f"{RED}Unknown persona '{arg}'. Available: {', '.join(personas)}{RESET}")
    sys.exit(1)


def build_turn_delta_tokens(
    tokenizer,
    mem_context: str,
    user_input: str,
    jailbreak_note: str = "",
) -> list[int]:
    """
    Token IDs for one user-turn delta — no BOS token.

    Prompt topology within [INST]:
      [Critical instruction]   ← only if jailbreak detected (p > 0.90)
      {jailbreak_note}

      [Memory context for this response]   ← only if mem_context non-empty
      {mem_context}

      {user_input}             ← always last, closest to generation boundary

    The absence of a BOS token is intentional: this delta is appended to an
    existing KV cache that already holds the static prefix (system_prompt +
    opening response). Mistral multi-turn format is:
        <s>[INST] … [/INST]resp</s>[INST] … [/INST]resp</s>…
    so each subsequent turn starts with [INST] (no <s>).
    """
    parts: list[str] = []
    if jailbreak_note:
        parts.append(f"[Critical instruction]\n{jailbreak_note}")
    if mem_context:
        parts.append(f"[Memory context for this response]\n{mem_context}")
    parts.append(user_input)

    content = "\n\n".join(parts)
    text = f"[INST] {content} [/INST]"
    return tokenizer.encode(text, add_special_tokens=False)


# ── Guard ─────────────────────────────────────────────────────────────────────

class Guard:
    BLOCK_THRESHOLD = 0.80

    def __init__(self):
        print(f"{DIM}Loading guard classifier...{RESET}", end=" ", flush=True)
        with open(CLF_PATH, "rb") as f:
            self._clf = pickle.load(f)
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("ready.")

    def encode(self, text: str):
        """Return embedding array of shape (1, dim).
        Call this once per turn and pass the result to both classify() and
        memory.retrieve_context() to avoid encoding the same text twice."""
        return self._embedder.encode([text])

    def classify(self, text: str, embedding=None) -> tuple[bool, float]:
        """Returns (blocked, p).
        Pass a pre-computed embedding (from encode()) to skip re-encoding."""
        kw_hit, _ = keyword_hit(text)
        emb = embedding if embedding is not None else self._embedder.encode([text])
        p = float(self._clf.predict_proba(emb)[0][1])
        blocked = kw_hit or p >= self.BLOCK_THRESHOLD
        return blocked, p


# ── Main loop ─────────────────────────────────────────────────────────────────

def chat_loop(
    model,
    tokenizer,
    guard: Guard,
    memory: HierarchicalMemory,
    system_prompt: str,
    npc_name: str,
    location: str,
    use_history: bool,
    temperature: float,
    rep_penalty: float,
    k_world: int,
    k_player: int,
    k_persona: int,
    k_conv: int,
    max_tokens: int = 160,
    timing: bool = False,
    prebaked_cache_file: "Path | None" = None,
    adapter_label: str = "base",
):
    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}{CYAN}{npc_name}{RESET}  ·  {location}")
    print(f"  {DIM}Ostwick, frontier town — type 'quit' to leave{RESET}")

    cache_mode = "prebaked" if prebaked_cache_file else "incremental"
    # adapter id is stamped onto the model object's path attribute when loaded
    # via mlx_lm; we read it back from the cache file when available, else
    # show "base"
    flags = [
        f"history={'on' if use_history else 'off'}",
        f"temp={temperature}",
        f"rep_penalty={rep_penalty}",
        f"memory=on (W={k_world}/L={k_player}/P={k_persona}/C={k_conv})",
        f"max_tokens={max_tokens}",
        f"kv_cache={cache_mode}",
        f"adapter={adapter_label}",
    ]
    if timing:
        flags.append("timing=on")
    print(f"  {DIM}Settings: {' · '.join(flags)}{RESET}")
    print(f"{BOLD}{'─'*56}{RESET}\n")

    sampler   = make_sampler(temp=temperature)
    logit_prs = make_logits_processors(repetition_penalty=rep_penalty)

    # ── Static prefix: build or load KV cache ────────────────────────────────
    opening_cue = "(The traveller approaches. Give a short in-character greeting.)"

    if prebaked_cache_file is not None:
        # ── γ path: load pre-baked cache from SSD ────────────────────────────
        # The cache already contains: system_prompt + all world_knowledge +
        # opening_cue + opening_response + EOS.  No prefill computation needed.
        print(f"{DIM}Loading pre-baked KV cache from {prebaked_cache_file.name}...{RESET}",
              end=" ", flush=True)
        cache, static_cache_len, prebaked_meta = load_cache(model, prebaked_cache_file)
        print(f"ready ({static_cache_len} tokens, "
              f"{prebaked_meta.get('world_facts_count', '?')} world facts baked in).")

        # Display the stored opening (generated during baking, no recompute)
        opening = prebaked_meta["opening"]
        print(f"{CYAN}{BOLD}[{npc_name}]{RESET} {opening}\n")

    else:
        # ── β path: live prefill — stream opening, cache fills during generation
        opening_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_prompt}\n\n{opening_cue}"}],
            tokenize=False, add_generation_prompt=True,
        )

        cache = make_prompt_cache(model)  # empty KV cache for all layers

        print(f"{CYAN}{BOLD}[{npc_name}]{RESET} ", end="", flush=True)
        opening_parts: list[str] = []
        for chunk in stream_generate(
            model, tokenizer,
            prompt=opening_input,
            prompt_cache=cache,
            max_tokens=80, sampler=sampler,
            logits_processors=logit_prs,
        ):
            text = chunk.text if hasattr(chunk, "text") else chunk
            print(text, end="", flush=True)
            opening_parts.append(text)
        print("\n")

        # Offset after static prefix (system + opening_cue + opening + EOS).
        # Used to trim the cache back to this point when --no-history is active.
        static_cache_len: int = cache[0].offset

    while True:
        try:
            raw = input(f"{YELLOW}You >{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}(You walk away.){RESET}")
            break

        if not raw:
            continue
        if raw.lower() in QUIT_WORDS:
            print(f"\n{DIM}(You walk away.){RESET}")
            break

        # ── Shared embedding: encode once, reuse for guard + retrieval ────────
        t_guard0 = time.perf_counter()
        embedding = guard.encode(raw)                          # ~190ms, shape (1, dim)
        blocked, p = guard.classify(raw, embedding=embedding)  # <1ms
        t_guard = time.perf_counter() - t_guard0
        p_tag = f"{DIM}[p={p:.2f}]{RESET}"

        # Jailbreak guard is injected into the delta context block (not system,
        # which is locked in the cache).
        if blocked and p >= JAILBREAK_THRESHOLD:
            jailbreak_note = JAILBREAK_GUARD.strip()
            tag = f"{RED}[jailbreak]{RESET}"
        elif blocked:
            jailbreak_note = ""
            tag = f"{DIM}[modern]{RESET}"
        else:
            jailbreak_note = ""
            tag = ""

        # ── Memory retrieval across all 4 layers (single shared embedding) ────
        t_mem0 = time.perf_counter()
        mem_context, mem_hits = memory.retrieve_context(
            raw,
            k_world=k_world, k_player=k_player,
            k_persona=k_persona, k_conv=k_conv,
            embedding=embedding,
        )
        t_mem = time.perf_counter() - t_mem0
        mem_tag = (
            f"{GREEN}[mem W{mem_hits['world']}/L{mem_hits['player']}"
            f"/P{mem_hits['persona']}/C{mem_hits['conv']}]{RESET}"
            if mem_context else ""
        )

        # ── Build delta: only the new user turn (~150 tokens, no BOS) ─────────
        delta_tokens = build_turn_delta_tokens(
            tokenizer, mem_context, raw, jailbreak_note=jailbreak_note
        )

        # ── Streaming generation — extend KV cache with delta + response ──────
        t_gen0 = time.perf_counter()
        print(f"\n{CYAN}{BOLD}[{npc_name}]{RESET} ", end="", flush=True)
        response_parts: list[str] = []
        t_first_token: float | None = None
        for chunk in stream_generate(
            model, tokenizer,
            prompt=delta_tokens,      # only new tokens; cache holds the rest
            prompt_cache=cache,       # KV states extended in-place
            max_tokens=max_tokens, sampler=sampler,
            logits_processors=logit_prs,
        ):
            # mlx_lm >= 0.21 yields GenerationResponse objects; older yields str
            text = chunk.text if hasattr(chunk, "text") else chunk
            if t_first_token is None:
                t_first_token = time.perf_counter() - t_gen0  # TTFT
            print(text, end="", flush=True)
            response_parts.append(text)
        response = "".join(response_parts).strip()
        t_gen = time.perf_counter() - t_gen0
        ttft = t_first_token or t_gen  # fallback if no tokens were produced

        cache_len = cache[0].offset
        if timing:
            timing_tag = (
                f"{DIM}[guard={t_guard*1000:.0f}ms"
                f" | mem={t_mem*1000:.0f}ms"
                f" | ttft={ttft*1000:.0f}ms"
                f" | gen={t_gen:.1f}s"
                f" | total={(t_guard+t_mem+t_gen):.1f}s"
                f" | cache={cache_len}tok]{RESET}"
            )
        else:
            timing_tag = f"{DIM}[ttft={ttft*1000:.0f}ms | {t_gen:.1f}s | cache={cache_len}tok]{RESET}"

        print(f"\n  {tag} {mem_tag} {p_tag} {timing_tag}\n")

        # ── Persist this turn to ChromaDB ─────────────────────────────────────
        memory.add_turn(raw, response)

        # ── Cache management ──────────────────────────────────────────────────
        if not use_history:
            # --no-history: trim the KV cache back to the static prefix so each
            # turn starts without any in-context conversation history.
            # ChromaDB semantic memory is unaffected.
            tokens_added = cache[0].offset - static_cache_len
            if tokens_added > 0:
                trim_prompt_cache(cache, tokens_added)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Talk to an RPG NPC with modular memory.")
    parser.add_argument("--persona", "-p", default=None,
                        help="Persona id or alias (e.g. 'marta', 'guard_roderick')")
    parser.add_argument("--no-history", dest="history", action="store_false",
                        help="Disable in-context sliding window (ChromaDB memory still active)")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear conversation history before starting (world knowledge is preserved)")
    parser.add_argument("--temp", type=float, default=0.75,
                        help="Sampling temperature (default: 0.75)")
    parser.add_argument("--rep-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1, off=1.0)")
    parser.add_argument("--k-world",   type=int, default=2,
                        help="L0 world_global facts per turn (default: 2)")
    parser.add_argument("--k-player",  type=int, default=2,
                        help="L_p player_lore facts per turn (default: 2)")
    parser.add_argument("--k-persona", type=int, default=2,
                        help="L3 persona-lore facts per turn (default: 2)")
    parser.add_argument("--k-conv",    type=int, default=3,
                        help="L4 past conversation turns per turn (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=160,
                        help="Max tokens to generate per turn (default: 160; try 80-120 for faster responses)")
    parser.add_argument("--timing", action="store_true",
                        help="Show per-stage latency breakdown (guard / mem / gen) each turn")
    parser.add_argument("--prebaked-cache", dest="prebaked_cache",
                        choices=["auto", "on", "off"], default="auto",
                        help="Use pre-baked KV cache (方案 γ): "
                             "'auto' = use if available (default), "
                             "'on' = require it (error if missing), "
                             "'off' = always use live β prefill")
    parser.add_argument("--adapter", default=None,
                        help="Path to a LoRA adapter directory (e.g. "
                             "outputs/adapters/v9_curated_v9). Omit to run "
                             "the base model (warning printed). γ caches are "
                             "looked up keyed on the adapter id.")
    parser.set_defaults(history=True)
    args = parser.parse_args()

    personas = load_personas()
    persona_id = resolve_persona(args.persona, personas) if args.persona else pick_persona(personas)

    persona_data   = personas[persona_id]
    system_prompt  = persona_data["system_prompt"].strip()
    world_knowledge = persona_data.get("world_knowledge", [])
    npc_name, location = PERSONA_DISPLAY[persona_id]

    # Guard
    guard = Guard()

    # Memory hierarchy (L0 + L_p + L3 + L4)
    print(f"{DIM}Initialising memory for {npc_name}...{RESET}", end=" ", flush=True)
    memory = HierarchicalMemory(npc_id=persona_id)

    if args.fresh:
        cleared = memory.clear_conv()
        print(f"\n  {YELLOW}--fresh: cleared {cleared} conversation turn(s). "
              f"Persona lore, world lore, player lore unchanged.{RESET}")

    # Seed L3 persona lore from YAML only on first run (no-op if populated)
    seeded = memory.seed_persona_lore(world_knowledge)
    if seeded:
        print(f"\n  {DIM}First run: seeded {len(world_knowledge)} persona-lore entries "
              f"from personas.yaml.{RESET}")

    print(
        f"ready ("
        f"world={memory.world.count()}, "
        f"player_lore={memory.player_lore.count()}, "
        f"persona={memory.persona_count()}, "
        f"conv={memory.conv_count()}"
        f")."
    )
    if memory.world.count() == 0:
        print(f"  {YELLOW}Warning: world_global is empty. "
              f"Run scripts/23_seed_world_knowledge.py.{RESET}")

    # Model + optional LoRA adapter
    adapter_id = adapter_id_from_path(args.adapter)
    if args.adapter:
        if not Path(args.adapter).exists():
            print(f"{RED}Adapter path not found: {args.adapter}{RESET}")
            sys.exit(1)
        print(f"{DIM}Loading {MODEL_ID} + adapter [{adapter_id}]...{RESET}",
              end=" ", flush=True)
        model, tokenizer = load(MODEL_ID, adapter_path=args.adapter)
    else:
        print(f"{YELLOW}No --adapter given; running base model "
              f"(persona is steered by system_prompt only).{RESET}")
        print(f"{DIM}Loading {MODEL_ID}...{RESET}", end=" ", flush=True)
        model, tokenizer = load(MODEL_ID)
    print("ready.\n")

    # ── Pre-baked cache resolution (方案 γ) ───────────────────────────────────
    # Caches are keyed on (npc, model, adapter) — different adapters need
    # different caches. is_valid() rejects an adapter mismatch automatically.
    prebaked_file: "Path | None" = None
    if args.prebaked_cache != "off":
        candidate = prebaked_path(persona_id, MODEL_ID, adapter_id=adapter_id)
        if cache_is_valid(candidate, persona_id, MODEL_ID, adapter_id=adapter_id):
            prebaked_file = candidate
            print(f"{DIM}Pre-baked cache found: {candidate.name}{RESET}")
        elif args.prebaked_cache == "on":
            print(f"{RED}--prebaked-cache on: no valid cache for "
                  f"{npc_name} (adapter={adapter_id}). Run:\n"
                  f"  python scripts/22_prebake_cache.py -p {persona_id}"
                  f"{' --adapter ' + args.adapter if args.adapter else ''}"
                  f"{RESET}")
            sys.exit(1)
        else:
            # auto — no cache available, fall through to β live prefill
            pass

    chat_loop(
        model, tokenizer, guard, memory,
        system_prompt, npc_name, location,
        use_history=args.history,
        temperature=args.temp,
        rep_penalty=args.rep_penalty,
        k_world=args.k_world,
        k_player=args.k_player,
        k_persona=args.k_persona,
        k_conv=args.k_conv,
        max_tokens=args.max_tokens,
        timing=args.timing,
        prebaked_cache_file=prebaked_file,
        adapter_label=adapter_id,
    )


if __name__ == "__main__":
    main()
