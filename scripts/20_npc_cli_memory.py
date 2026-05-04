"""
20_npc_cli_memory.py — Interactive NPC CLI with Modular Memory

Extension of 19_npc_cli.py that adds ChromaDB-backed modular memory:
  - World Knowledge: NPC-specific facts retrieved via semantic similarity
  - Conversational Memory: past dialogue turns retrieved by relevance

Pipeline:
  User input
    → Tier 1: keyword check  (instant)
    → Tier 2: embedding score via guard_model.pkl
    → p > 0.90 → inject JAILBREAK_GUARD into system prompt
    → memory.retrieve_context(user_input) → inject into system prompt
    → MLX 4-bit Mistral-7B inference → NPC response
    → memory.add_turn(user_input, response) → persist to ChromaDB

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
    --no-history      Disable in-context sliding window (ChromaDB memory still active)
    --fresh           Clear this NPC's conversation history (_conv) before starting.
                      World knowledge (_world) is never touched by --fresh.
    --temp FLOAT      Sampling temperature, default 0.75
    --rep-penalty FLOAT  Repetition penalty, default 1.1
    --n-world INT     World knowledge facts to inject per turn (default 3)
    --n-conv INT      Past turns to retrieve per turn (default 2)
    --max-tokens INT  Max tokens to generate per turn (default 160)
    --timing          Show per-stage latency: guard / mem / gen each turn

Optimisations active by default:
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

from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from src.memory_module import ModularMemory

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


def build_prompt(
    tokenizer,
    system: str,
    memory_context: str,
    history: list[dict],
    new_user: str,
) -> str:
    """
    Build a multi-turn prompt.

    memory_context is injected into the system block (after persona, before
    conversation) so the model can reference relevant facts when composing
    each response.

    Mistral doesn't support "system" role — inject system into the FIRST user turn.
    """
    # Compose effective system: persona + retrieved memory context
    effective_system = system
    if memory_context:
        effective_system = f"{system}\n\n[Memory context for this response]\n{memory_context}"

    messages = []
    for i, turn in enumerate(history):
        if turn["role"] == "user" and i == 0:
            messages.append({"role": "user", "content": f"{effective_system}\n\n{turn['content']}"})
        else:
            messages.append(turn)

    if not messages:
        messages.append({"role": "user", "content": f"{effective_system}\n\n{new_user}"})
    else:
        messages.append({"role": "user", "content": new_user})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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
    memory: ModularMemory,
    system_prompt: str,
    npc_name: str,
    location: str,
    use_history: bool,
    temperature: float,
    rep_penalty: float,
    n_world: int,
    n_conv: int,
    max_tokens: int = 160,
    timing: bool = False,
):
    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}{CYAN}{npc_name}{RESET}  ·  {location}")
    print(f"  {DIM}Ostwick, frontier town — type 'quit' to leave{RESET}")

    flags = [
        f"history={'on' if use_history else 'off'}",
        f"temp={temperature}",
        f"rep_penalty={rep_penalty}",
        f"memory=on (world={n_world}, conv={n_conv})",
        f"max_tokens={max_tokens}",
    ]
    if timing:
        flags.append("timing=on")
    print(f"  {DIM}Settings: {' · '.join(flags)}{RESET}")
    print(f"{BOLD}{'─'*56}{RESET}\n")

    history: list[dict] = []
    sampler   = make_sampler(temp=temperature)
    logit_prs = make_logits_processors(repetition_penalty=rep_penalty)

    # Opening greeting (no memory retrieval — nothing stored yet)
    opening_user = "(The traveller approaches. Give a short in-character greeting.)"
    opening_prompt = build_prompt(tokenizer, system_prompt, "", [], opening_user)
    opening = generate(
        model, tokenizer, prompt=opening_prompt,
        max_tokens=80, sampler=sampler,
        logits_processors=logit_prs, verbose=False,
    ).strip()
    print(f"{CYAN}{BOLD}[{npc_name}]{RESET} {opening}\n")

    if use_history:
        history.append({"role": "user",      "content": opening_user})
        history.append({"role": "assistant", "content": opening})

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

        # ── Shared embedding: encode once, reuse for guard + retrieval ──────────
        t_guard0 = time.perf_counter()
        embedding = guard.encode(raw)                          # ~190ms, shape (1, dim)
        blocked, p = guard.classify(raw, embedding=embedding)  # <1ms
        t_guard = time.perf_counter() - t_guard0
        p_tag = f"{DIM}[p={p:.2f}]{RESET}"

        if blocked and p >= JAILBREAK_THRESHOLD:
            effective_system = system_prompt + JAILBREAK_GUARD
            tag = f"{RED}[jailbreak]{RESET}"
        elif blocked:
            effective_system = system_prompt
            tag = f"{DIM}[modern]{RESET}"
        else:
            effective_system = system_prompt
            tag = ""

        # Retrieve memory context (reuses embedding — no second encode())
        t_mem0 = time.perf_counter()
        mem_context = memory.retrieve_context(
            raw, n_world=n_world, n_conv=n_conv, embedding=embedding
        )
        t_mem = time.perf_counter() - t_mem0
        mem_tag = f"{GREEN}[mem]{RESET}" if mem_context else ""

        # Build prompt
        current_history = history if use_history else []
        prompt = build_prompt(tokenizer, effective_system, mem_context, current_history, raw)

        # Streaming generation — tokens appear as they are produced
        t_gen0 = time.perf_counter()
        print(f"\n{CYAN}{BOLD}[{npc_name}]{RESET} ", end="", flush=True)
        response_parts: list[str] = []
        t_first_token: float | None = None
        for chunk in stream_generate(
            model, tokenizer, prompt=prompt,
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

        if timing:
            timing_tag = (
                f"{DIM}[guard={t_guard*1000:.0f}ms"
                f" | mem={t_mem*1000:.0f}ms"
                f" | ttft={ttft*1000:.0f}ms"
                f" | gen={t_gen:.1f}s"
                f" | total={(t_guard+t_mem+t_gen):.1f}s]{RESET}"
            )
        else:
            timing_tag = f"{DIM}[ttft={ttft*1000:.0f}ms | {t_gen:.1f}s]{RESET}"

        print(f"\n  {tag} {mem_tag} {p_tag} {timing_tag}\n")

        # Persist this turn to ChromaDB
        memory.add_turn(raw, response)

        # Update in-context history
        if use_history:
            history.append({"role": "user",      "content": raw})
            history.append({"role": "assistant", "content": response})


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
    parser.add_argument("--n-world", type=int, default=3,
                        help="World knowledge facts to retrieve per turn (default: 3)")
    parser.add_argument("--n-conv", type=int, default=2,
                        help="Past conversation turns to retrieve per turn (default: 2)")
    parser.add_argument("--max-tokens", type=int, default=160,
                        help="Max tokens to generate per turn (default: 160; try 80-120 for faster responses)")
    parser.add_argument("--timing", action="store_true",
                        help="Show per-stage latency breakdown (guard / mem / gen) each turn")
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

    # Memory
    print(f"{DIM}Initialising memory for {npc_name}...{RESET}", end=" ", flush=True)
    memory = ModularMemory(npc_id=persona_id)

    if args.fresh:
        cleared = memory.clear_conv()
        print(f"\n  {YELLOW}--fresh: cleared {cleared} conversation turn(s). World knowledge unchanged.{RESET}")

    # Seed world knowledge from YAML only on first run (no-op if already populated)
    seeded = memory.seed_world_knowledge(world_knowledge)
    if seeded:
        print(f"\n  {DIM}First run: seeded {len(world_knowledge)} world knowledge entries from personas.yaml.{RESET}")

    print(f"ready ({memory.world_count()} world facts, {memory.conv_count()} stored turns).")

    # Model
    print(f"{DIM}Loading {MODEL_ID}...{RESET}", end=" ", flush=True)
    model, tokenizer = load(MODEL_ID)
    print("ready.\n")

    chat_loop(
        model, tokenizer, guard, memory,
        system_prompt, npc_name, location,
        use_history=args.history,
        temperature=args.temp,
        rep_penalty=args.rep_penalty,
        n_world=args.n_world,
        n_conv=args.n_conv,
        max_tokens=args.max_tokens,
        timing=args.timing,
    )


if __name__ == "__main__":
    main()
