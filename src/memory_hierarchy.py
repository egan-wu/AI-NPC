"""
src/memory_hierarchy.py — Multi-Layer NPC Memory  (Phase 6)

Composes four ChromaDB collections into a single retrieval surface so any NPC
can answer with a coherent blend of shared world lore, the player's deeds,
their own background, and the current conversation.

Layer  Collection            Scope              Lifecycle             Maintained by
─────  ────────────────────  ─────────────────  ────────────────────  ────────────────────────────
L0     world_global          all NPCs           static (immutable)    23_seed_world_knowledge.py
L_p    player_lore           all NPCs           runtime, growing      24_manage_player_lore.py
L3     {npc_id}_persona      single NPC         static per NPC        21_manage_persona_lore.py
L4     {npc_id}_conv         single NPC         per session           20_npc_cli_memory.py (auto)

Retrieval format (whitespace-trimmed sections, omitted if empty):

    [World — common knowledge]
    - …
    - …

    [About the traveller]
    - …

    [About me]
    - …

    [Recent conversation]
    Player: …
    NPC: …

KV-cache compatibility:
    L0 + L3 are static and bake into the γ pre-baked cache (22_prebake_cache.py).
    L_p + L4 are dynamic and ride in the per-turn β delta (20_npc_cli_memory.py).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import uuid

import chromadb
from chromadb.utils import embedding_functions

from src.memory_module import ModularMemory


_DEFAULT_DB_DIR = Path(__file__).parent.parent / "outputs" / "chroma_db"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton client/embedder to avoid re-instantiating on every memory object
_client_cache: dict[str, chromadb.PersistentClient] = {}
_ef_cache: dict[str, "embedding_functions.SentenceTransformerEmbeddingFunction"] = {}


def _get_client(db_dir: str) -> chromadb.PersistentClient:
    if db_dir not in _client_cache:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        _client_cache[db_dir] = chromadb.PersistentClient(path=db_dir)
    return _client_cache[db_dir]


def _get_ef(model_name: str):
    if model_name not in _ef_cache:
        _ef_cache[model_name] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    return _ef_cache[model_name]


# ── L0: shared world knowledge ────────────────────────────────────────────────

class WorldKnowledgeStore:
    """CRUD for the shared `world_global` collection (L0)."""

    COLLECTION = "world_global"

    def __init__(self, db_dir: Path | str | None = None,
                 embedding_model: str = _EMBEDDING_MODEL):
        client = _get_client(str(db_dir or _DEFAULT_DB_DIR))
        self._col = client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=_get_ef(embedding_model)
        )

    def count(self) -> int:
        return self._col.count()

    def list_all(self) -> list[dict]:
        if self._col.count() == 0:
            return []
        result = self._col.get()
        pairs = list(zip(result["ids"], result["documents"]))
        pairs.sort(key=lambda x: x[0])
        return [{"id": i, "text": d} for i, d in pairs]

    def all_texts(self) -> list[str]:
        return [e["text"] for e in self.list_all()]

    def replace_all(self, facts: List[str]) -> int:
        """Drop everything in the collection and reinsert the supplied facts.

        Used by 23_seed_world_knowledge.py so the YAML file stays the source of
        truth — edits in YAML always win, deletes propagate.
        Returns the number of facts written.
        """
        if self._col.count() > 0:
            self._col.delete(ids=self._col.get()["ids"])
        if not facts:
            return 0
        ids = [f"wg_{i:03d}" for i in range(len(facts))]
        self._col.add(documents=facts, ids=ids)
        return len(facts)

    def query(self, embedding=None, query_text: str | None = None,
              n_results: int = 2) -> list[str]:
        if self._col.count() == 0:
            return []
        n = min(n_results, self._col.count())
        kwargs = _query_kwargs(embedding, query_text)
        result = self._col.query(**kwargs, n_results=n)
        return result["documents"][0] if result["documents"] else []


# ── L_p: player lore (shared, runtime-growing) ────────────────────────────────

class PlayerLoreStore:
    """CRUD for the shared `player_lore` collection (L_p).

    Facts are written by 24_manage_player_lore.py whenever the player does
    something every NPC would plausibly hear about (slew a dragon, etc.).
    Each entry carries a `timestamp` metadata field for future recency ranking.
    """

    COLLECTION = "player_lore"

    def __init__(self, db_dir: Path | str | None = None,
                 embedding_model: str = _EMBEDDING_MODEL):
        client = _get_client(str(db_dir or _DEFAULT_DB_DIR))
        self._col = client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=_get_ef(embedding_model)
        )

    def count(self) -> int:
        return self._col.count()

    def list_all(self) -> list[dict]:
        if self._col.count() == 0:
            return []
        result = self._col.get(include=["documents", "metadatas"])
        rows = list(zip(result["ids"], result["documents"], result["metadatas"]))
        # Newest first
        rows.sort(key=lambda r: (r[2] or {}).get("timestamp", ""), reverse=True)
        return [
            {"id": i, "text": d, "timestamp": (m or {}).get("timestamp", "")}
            for i, d, m in rows
        ]

    def add(self, text: str, timestamp: str | None = None) -> str:
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        new_id = f"plr_{uuid.uuid4().hex[:10]}"
        self._col.add(documents=[text], ids=[new_id], metadatas=[{"timestamp": ts}])
        return new_id

    def remove(self, entry_id: str) -> None:
        self._col.delete(ids=[entry_id])

    def clear(self) -> int:
        n = self._col.count()
        if n > 0:
            self._col.delete(ids=self._col.get()["ids"])
        return n

    def query(self, embedding=None, query_text: str | None = None,
              n_results: int = 2) -> list[str]:
        if self._col.count() == 0:
            return []
        n = min(n_results, self._col.count())
        kwargs = _query_kwargs(embedding, query_text)
        result = self._col.query(**kwargs, n_results=n)
        return result["documents"][0] if result["documents"] else []


# ── Hierarchical aggregator ───────────────────────────────────────────────────

def _query_kwargs(embedding, query_text):
    """Prefer a pre-computed embedding when one is supplied (saves an encode)."""
    if embedding is not None:
        vec = embedding[0] if hasattr(embedding, "ndim") and embedding.ndim == 2 \
              else embedding
        return {"query_embeddings": [vec.tolist() if hasattr(vec, "tolist") else list(vec)]}
    return {"query_texts": [query_text or ""]}


class HierarchicalMemory:
    """Façade over all four memory layers for the chat loop.

    Use this from `20_npc_cli_memory.py`. Per-layer stores are still accessible
    via `.world`, `.player_lore`, `.npc` if a script needs CRUD on a single layer.
    """

    def __init__(
        self,
        npc_id: str,
        db_dir: Path | str | None = None,
        embedding_model: str = _EMBEDDING_MODEL,
    ):
        self.npc_id = npc_id
        self.world        = WorldKnowledgeStore(db_dir, embedding_model)
        self.player_lore  = PlayerLoreStore(db_dir, embedding_model)
        self.npc          = ModularMemory(
            npc_id=npc_id, db_dir=db_dir, embedding_model=embedding_model,
        )

    # ── Aggregated retrieval ──────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        k_world:   int = 2,
        k_player:  int = 2,
        k_persona: int = 2,
        k_conv:    int = 3,
        embedding = None,
    ) -> tuple[str, dict[str, int]]:
        """
        Query all four layers and return (formatted_string, hit_counts).

        Each layer is queried independently so per-layer top-k can be tuned.
        Hit counts are returned for `--timing` instrumentation
        (e.g. `mem=W2/L1/P2/C3`).
        """
        sections: list[str] = []
        hits = {"world": 0, "player": 0, "persona": 0, "conv": 0}

        if k_world > 0:
            docs = self.world.query(embedding=embedding, query_text=query,
                                    n_results=k_world)
            if docs:
                sections.append(
                    "[World — common knowledge]\n" + "\n".join(f"- {d}" for d in docs)
                )
                hits["world"] = len(docs)

        if k_player > 0:
            docs = self.player_lore.query(embedding=embedding, query_text=query,
                                          n_results=k_player)
            if docs:
                sections.append(
                    "[About the traveller]\n" + "\n".join(f"- {d}" for d in docs)
                )
                hits["player"] = len(docs)

        # Persona + conv: re-use the existing ModularMemory.retrieve_context
        # so its labelled output ("About me:", "Related past exchanges:") stays
        # the canonical wording. We split the result so the section header
        # styles stay consistent here.
        npc_block = self.npc.retrieve_context(
            query, n_persona=k_persona, n_conv=k_conv, embedding=embedding,
        )
        if npc_block:
            sections.append(npc_block)
            # Approximate hit counts for instrumentation
            hits["persona"] = min(k_persona, self.npc.persona_count())
            hits["conv"]    = min(k_conv,    self.npc.conv_count())

        return "\n\n".join(sections), hits

    # ── Convenience passthroughs used by chat_loop ────────────────────────────

    def add_turn(self, player_msg: str, npc_msg: str) -> None:
        self.npc.add_turn(player_msg, npc_msg)

    def clear_conv(self) -> int:
        return self.npc.clear_conv()

    def conv_count(self) -> int:
        return self.npc.conv_count()

    def seed_persona_lore(self, knowledge_list: List[str]) -> bool:
        return self.npc.seed_persona_lore(knowledge_list)

    def persona_count(self) -> int:
        return self.npc.persona_count()

    # ── Static-prefix facts for γ pre-bake ────────────────────────────────────

    def get_static_facts(self) -> dict[str, list[str]]:
        """
        Return all facts that should be baked into the γ static prefix.

        Returns a dict so the bake script can label each section in the prompt
        ("[World]" vs "[About me]") and so cache_utils.metadata can store the
        per-layer fact counts for invalidation.
        """
        return {
            "world":   self.world.all_texts(),
            "persona": self.npc.persona_all_texts(),
        }
