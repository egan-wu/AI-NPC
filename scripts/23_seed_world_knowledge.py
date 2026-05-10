"""
23_seed_world_knowledge.py — Seed the L0 shared world-lore collection.

Loads configs/world_knowledge.yaml and writes every entry into the ChromaDB
`world_global` collection. Idempotent: re-running drops the existing rows
and re-inserts so the YAML file stays the single source of truth (edits and
deletes propagate cleanly).

When to re-run
--------------
- After editing configs/world_knowledge.yaml
- After deleting outputs/chroma_db/

NOTE: After re-seeding, also re-bake any γ pre-baked caches that depend on
L0 contents:  python scripts/22_prebake_cache.py --all --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from src.memory_hierarchy import WorldKnowledgeStore

CONFIG_PATH = _root / "configs" / "world_knowledge.yaml"

DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def load_facts() -> list[str]:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    facts = cfg.get("facts", []) if isinstance(cfg, dict) else []
    # Normalise: strip whitespace, drop empties
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed L0 world_global from YAML.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written, don't touch the DB.")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print(f"{RED}Missing {CONFIG_PATH.relative_to(_root)}{RESET}")
        sys.exit(1)

    facts = load_facts()
    print(f"Loaded {len(facts)} fact(s) from "
          f"{DIM}{CONFIG_PATH.relative_to(_root)}{RESET}")

    if args.dry_run:
        for i, f in enumerate(facts, 1):
            print(f"  {i:>3}. {f}")
        print(f"\n{YELLOW}--dry-run: nothing written.{RESET}")
        return

    store = WorldKnowledgeStore()
    prev_count = store.count()
    n_written = store.replace_all(facts)

    print(f"{GREEN}✓ world_global: {prev_count} → {n_written} entries.{RESET}")
    if prev_count > 0:
        print(f"{DIM}Re-bake γ caches if any:  python scripts/22_prebake_cache.py --all --force{RESET}")


if __name__ == "__main__":
    main()
