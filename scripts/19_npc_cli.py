"""
19_npc_cli.py — Interactive NPC CLI

Talk to an RPG NPC in your terminal using local MLX inference.

Pipeline:
  User input
    → Tier 1: keyword check  (instant)
    → Tier 2: embedding score via pre-trained guard_model.pkl
    → p > 0.90 → inject JAILBREAK_GUARD into system prompt
    → MLX 4-bit Mistral-7B inference → NPC response

Usage:
    source .venv/bin/activate
    python scripts/19_npc_cli.py                        # pick persona interactively
    python scripts/19_npc_cli.py -p marta               # short alias ok
    python scripts/19_npc_cli.py -p marta --no-history  # disable conversation history
    python scripts/19_npc_cli.py -p marta --temp 0.9 --rep-penalty 1.15

Key parameters:
    --no-history      Disable conversation memory (each turn is independent)
    --temp FLOAT      Sampling temperature, default 0.75  (higher = more varied)
    --rep-penalty FLOAT  Repetition penalty, default 1.1  (higher = less repetition)

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

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from src.runtime_config import get_config

# ── Constants (paths only; tunables come from runtime_config) ─────────────────
_CFG          = get_config()
MODEL_ID      = _CFG.model_id
CLF_PATH      = _root / "outputs" / "guard_model.pkl"
PERSONAS_PATH = _root / "configs" / "personas.yaml"

JAILBREAK_THRESHOLD = _CFG.guard.jailbreak_threshold
JAILBREAK_GUARD = "\n\n" + _CFG.guard.jailbreak_guard_text

PERSONA_DISPLAY = _CFG.persona_display

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


def build_prompt(tokenizer, system: str, history: list[dict], new_user: str) -> str:
    """
    Build a multi-turn prompt.
    history: list of {"role": "user"|"assistant", "content": str}

    Mistral doesn't support "system" role — inject system into the FIRST user turn.
    """
    messages = []
    for i, turn in enumerate(history):
        if turn["role"] == "user" and i == 0:
            # Prepend system to the very first user turn
            messages.append({"role": "user", "content": f"{system}\n\n{turn['content']}"})
        else:
            messages.append(turn)

    # Add the new user message
    if not messages:
        # First ever message
        messages.append({"role": "user", "content": f"{system}\n\n{new_user}"})
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

    def classify(self, text: str) -> tuple[bool, float]:
        """Returns (blocked, p)."""
        kw_hit, _ = keyword_hit(text)
        emb = self._embedder.encode([text])
        p = float(self._clf.predict_proba(emb)[0][1])
        blocked = kw_hit or p >= self.BLOCK_THRESHOLD
        return blocked, p


# ── Main loop ─────────────────────────────────────────────────────────────────

def chat_loop(
    model,
    tokenizer,
    guard: Guard,
    system_prompt: str,
    npc_name: str,
    location: str,
    use_history: bool,
    temperature: float,
    rep_penalty: float,
):
    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}{CYAN}{npc_name}{RESET}  ·  {location}")
    print(f"  {DIM}Ostwick, frontier town — type 'quit' to leave{RESET}")

    # Show active settings
    flags = []
    flags.append(f"history={'on' if use_history else 'off'}")
    flags.append(f"temp={temperature}")
    flags.append(f"rep_penalty={rep_penalty}")
    print(f"  {DIM}Settings: {' · '.join(flags)}{RESET}")
    print(f"{BOLD}{'─'*56}{RESET}\n")

    # Conversation history (only used when use_history=True)
    history: list[dict] = []

    sampler   = make_sampler(temp=temperature)
    logit_prs = make_logits_processors(repetition_penalty=rep_penalty)

    # Opening greeting
    opening_user = "(The traveller approaches. Give a short in-character greeting.)"
    opening_prompt = build_prompt(tokenizer, system_prompt, [], opening_user)
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

        # Guard classification
        blocked, p = guard.classify(raw)
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

        # Build prompt (with or without history)
        current_history = history if use_history else []
        prompt = build_prompt(tokenizer, effective_system, current_history, raw)

        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=160, sampler=sampler,
            logits_processors=logit_prs, verbose=False,
        ).strip()
        elapsed = f"{DIM}[{time.time() - t0:.1f}s]{RESET}"

        print(f"\n{CYAN}{BOLD}[{npc_name}]{RESET} {response}  {tag} {p_tag} {elapsed}\n")

        # Update history
        if use_history:
            history.append({"role": "user",      "content": raw})
            history.append({"role": "assistant", "content": response})


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Talk to an RPG NPC locally.")
    parser.add_argument("--persona", "-p", default=None,
                        help="Persona id or alias (e.g. 'marta', 'guard_roderick')")
    parser.add_argument("--no-history", dest="history", action="store_false",
                        help="Disable conversation memory (each turn is independent)")
    parser.add_argument("--temp", type=float, default=0.75,
                        help="Sampling temperature (default: 0.75)")
    parser.add_argument("--rep-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1, off=1.0)")
    parser.set_defaults(history=True)
    args = parser.parse_args()

    personas = load_personas()
    persona_id = resolve_persona(args.persona, personas) if args.persona else pick_persona(personas)

    persona_data  = personas[persona_id]
    system_prompt = persona_data["system_prompt"].strip()
    npc_name, location = PERSONA_DISPLAY[persona_id]

    guard = Guard()

    print(f"{DIM}Loading {MODEL_ID}...{RESET}", end=" ", flush=True)
    model, tokenizer = load(MODEL_ID)
    print("ready.\n")

    chat_loop(
        model, tokenizer, guard,
        system_prompt, npc_name, location,
        use_history=args.history,
        temperature=args.temp,
        rep_penalty=args.rep_penalty,
    )


if __name__ == "__main__":
    main()
