"""
24_manage_player_lore.py — Player Lore Manager  (L_p)

CRUD for the shared `player_lore` ChromaDB collection. Every NPC retrieves
from this collection at inference time, so a fact added here is "known" by
every NPC after the next turn (no rebake needed — L_p is dynamic).

Usage
-----
    source .venv/bin/activate

    # Add a deed — every NPC will know about it
    python scripts/24_manage_player_lore.py add "Slew the Greycrest dragon"
    python scripts/24_manage_player_lore.py add "Spared goblin chief Grakk"

    # List (newest first)
    python scripts/24_manage_player_lore.py list

    # Remove by index (1-based, matches `list` output)
    python scripts/24_manage_player_lore.py remove 1

    # Wipe all player lore (e.g. starting a new save)
    python scripts/24_manage_player_lore.py clear

    # Interactive menu
    python scripts/24_manage_player_lore.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from src.memory_hierarchy import PlayerLoreStore

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
GREEN  = "\033[92m"


def short_date(ts: str) -> str:
    """ISO timestamp → 'YYYY-MM-DD' for display."""
    return ts[:10] if ts else "—"


def cmd_list(store: PlayerLoreStore) -> list[dict]:
    entries = store.list_all()
    print(f"\n{BOLD}Player lore ({len(entries)} entries, newest first):{RESET}")
    if not entries:
        print(f"  {DIM}(empty){RESET}")
    for i, e in enumerate(entries, 1):
        print(f"  {CYAN}{i:>3}.{RESET} [{DIM}{short_date(e['timestamp'])}{RESET}] {e['text']}")
    print()
    return entries


def cmd_add(store: PlayerLoreStore, text: str) -> None:
    text = text.strip()
    if not text:
        print(f"  {RED}Empty fact — nothing added.{RESET}")
        return
    new_id = store.add(text)
    print(f"  {GREEN}✓ Added [{new_id}]:{RESET} {text}")


def cmd_remove(store: PlayerLoreStore, idx: int) -> None:
    entries = store.list_all()
    if not (1 <= idx <= len(entries)):
        print(f"  {RED}Index {idx} out of range (1..{len(entries)}).{RESET}")
        return
    target = entries[idx - 1]
    store.remove(target["id"])
    print(f"  {GREEN}✓ Removed:{RESET} {target['text']}")


def cmd_clear(store: PlayerLoreStore) -> None:
    n = store.count()
    if n == 0:
        print(f"  {DIM}Already empty.{RESET}")
        return
    confirm = input(f"  {YELLOW}Clear all {n} player lore entries? (y/N) > {RESET}").strip().lower()
    if confirm != "y":
        print(f"  {DIM}Cancelled.{RESET}")
        return
    n_cleared = store.clear()
    print(f"  {GREEN}✓ Cleared {n_cleared} entries.{RESET}")


# ── Interactive ────────────────────────────────────────────────────────────────

def interactive(store: PlayerLoreStore) -> None:
    print(f"\n{BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}Player Lore Manager{RESET}")
    print(f"  {DIM}Commands: (l)ist  (a)dd <text>  (r)emove <n>  (c)lear  (q)uit{RESET}")
    print(f"{BOLD}{'─'*56}{RESET}")
    cmd_list(store)

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

        if cmd in ("q", "quit", "exit"):
            print(f"{DIM}Bye.{RESET}")
            break
        elif cmd in ("l", "list"):
            cmd_list(store)
        elif cmd in ("a", "add"):
            if not rest:
                print(f"  {DIM}Enter text (blank cancels):{RESET}")
                rest = input("  > ").strip()
            cmd_add(store, rest)
        elif cmd in ("r", "remove", "rm", "d", "delete"):
            try:
                idx = int(rest)
            except ValueError:
                print(f"  {RED}Usage: r <number>{RESET}")
                continue
            cmd_remove(store, idx)
        elif cmd in ("c", "clear"):
            cmd_clear(store)
        else:
            print(f"  {RED}Unknown: '{cmd}'.  Try l / a / r <n> / c / q.{RESET}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Manage shared player lore (L_p).")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all entries (newest first)")

    p_add = sub.add_parser("add", help="Add a new player lore fact")
    p_add.add_argument("text", nargs="+", help="Fact text (quoted or bare words)")

    p_rm = sub.add_parser("remove", help="Remove entry by 1-based index")
    p_rm.add_argument("index", type=int)

    sub.add_parser("clear", help="Wipe all player lore")

    args = parser.parse_args()
    store = PlayerLoreStore()

    if args.cmd is None:
        interactive(store)
    elif args.cmd == "list":
        cmd_list(store)
    elif args.cmd == "add":
        cmd_add(store, " ".join(args.text))
    elif args.cmd == "remove":
        cmd_remove(store, args.index)
    elif args.cmd == "clear":
        cmd_clear(store)


if __name__ == "__main__":
    main()
