"""
21_manage_world_knowledge.py — Interactive World Knowledge Manager

Manage the persistent world knowledge (_world ChromaDB collection) for any NPC.
World knowledge survives --fresh resets; only conversation history (_conv) is
cleared by --fresh.

Usage:
    source .venv/bin/activate
    python scripts/21_manage_world_knowledge.py             # pick persona interactively
    python scripts/21_manage_world_knowledge.py -p marta    # short alias ok

Commands (interactive menu):
    l          List all entries
    a          Add a new entry
    e <n>      Edit entry number <n>
    d <n>      Delete entry number <n>
    q / quit   Exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from src.memory_module import ModularMemory

PERSONAS_PATH = _root / "configs" / "personas.yaml"

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
GREEN  = "\033[92m"

PERSONA_DISPLAY = {
    "innkeeper_marta":  ("Marta",    "Stag & Thistle Inn"),
    "merchant_garrick": ("Garrick",  "Travelling Merchant"),
    "guard_roderick":   ("Roderick", "Captain of the Watch"),
    "child_lily":       ("Lily",     "Baker's Daughter"),
    "hermit_wenric":    ("Wenric",   "Sage Hermit of Greycrest"),
}


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


def print_entries(entries: list[dict]) -> None:
    if not entries:
        print(f"  {DIM}(no entries){RESET}")
        return
    for i, e in enumerate(entries, 1):
        print(f"  {CYAN}{i:>3}.{RESET} [{DIM}{e['id']}{RESET}] {e['text']}")


def parse_index(raw: str, entries: list[dict]) -> int | None:
    """Parse a 1-based index from user input. Returns None if invalid."""
    try:
        n = int(raw)
        if 1 <= n <= len(entries):
            return n - 1   # 0-based
    except ValueError:
        pass
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def manage_loop(memory: ModularMemory, npc_name: str) -> None:
    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  World Knowledge Manager — {BOLD}{CYAN}{npc_name}{RESET}")
    print(f"  {DIM}Commands: (l)ist  (a)dd  (e)dit <n>  (d)elete <n>  (q)uit{RESET}")
    print(f"{BOLD}{'─'*56}{RESET}\n")

    # Show list on entry
    entries = memory.world_list()
    print_entries(entries)
    print()

    while True:
        try:
            raw = input(f"{YELLOW}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not raw:
            continue

        cmd = raw.split()[0].lower()
        rest = raw[len(cmd):].strip()

        # ── quit ──────────────────────────────────────────────────────────────
        if cmd in ("q", "quit", "exit"):
            print(f"{DIM}Bye.{RESET}")
            break

        # ── list ──────────────────────────────────────────────────────────────
        elif cmd in ("l", "list"):
            entries = memory.world_list()
            print()
            print_entries(entries)
            print()

        # ── add ───────────────────────────────────────────────────────────────
        elif cmd in ("a", "add"):
            if rest:
                text = rest
            else:
                print(f"  {DIM}Enter new entry (blank to cancel):{RESET}")
                text = input("  > ").strip()
            if not text:
                print(f"  {DIM}Cancelled.{RESET}")
                continue
            new_id = memory.world_add(text)
            print(f"  {GREEN}✓ Added [{new_id}]:{RESET} {text}\n")
            entries = memory.world_list()

        # ── edit ──────────────────────────────────────────────────────────────
        elif cmd in ("e", "edit"):
            entries = memory.world_list()
            idx = parse_index(rest, entries)
            if idx is None:
                print(f"  {RED}Usage: e <number>   (e.g. 'e 3'){RESET}")
                continue
            entry = entries[idx]
            print(f"  {DIM}Current [{entry['id']}]:{RESET} {entry['text']}")
            print(f"  {DIM}New text (blank to cancel):{RESET}")
            new_text = input("  > ").strip()
            if not new_text:
                print(f"  {DIM}Cancelled.{RESET}")
                continue
            memory.world_update(entry["id"], new_text)
            print(f"  {GREEN}✓ Updated [{entry['id']}]:{RESET} {new_text}\n")
            entries = memory.world_list()

        # ── delete ────────────────────────────────────────────────────────────
        elif cmd in ("d", "delete", "del", "rm"):
            entries = memory.world_list()
            idx = parse_index(rest, entries)
            if idx is None:
                print(f"  {RED}Usage: d <number>   (e.g. 'd 2'){RESET}")
                continue
            entry = entries[idx]
            print(f"  {YELLOW}Delete [{entry['id']}]: {entry['text'][:70]}{RESET}")
            confirm = input("  Confirm? (y/N) > ").strip().lower()
            if confirm == "y":
                memory.world_delete(entry["id"])
                print(f"  {GREEN}✓ Deleted.{RESET}\n")
                entries = memory.world_list()
            else:
                print(f"  {DIM}Cancelled.{RESET}")

        else:
            print(f"  {RED}Unknown command '{cmd}'. "
                  f"Try: l / a / e <n> / d <n> / q{RESET}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manage world knowledge for an NPC (ChromaDB _world collection)."
    )
    parser.add_argument("--persona", "-p", default=None,
                        help="Persona id or alias (e.g. 'marta', 'guard_roderick')")
    args = parser.parse_args()

    personas = load_personas()
    persona_id = resolve_persona(args.persona, personas) if args.persona else pick_persona(personas)
    npc_name, _ = PERSONA_DISPLAY[persona_id]

    # Seed from YAML if this NPC's world collection is still empty
    persona_data    = personas[persona_id]
    world_knowledge = persona_data.get("world_knowledge", [])

    memory = ModularMemory(npc_id=persona_id)
    seeded = memory.seed_world_knowledge(world_knowledge)
    if seeded:
        print(f"{DIM}First run: seeded {len(world_knowledge)} entries from personas.yaml.{RESET}")

    manage_loop(memory, npc_name)


if __name__ == "__main__":
    main()
