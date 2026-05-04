"""
src/memory_module.py — Modular NPC Memory (ChromaDB-backed)

Adapted from vendor/AI-NPC-Personality/npc_system/memory_module.py.
Original design: "Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue
on Consumer Hardware".

Key changes from upstream:
- Removed TinyLlama / HuggingFace inference coupling (we use MLX Mistral-7B)
- retrieve_context() returns a plain string ready for prompt injection
- ChromaDB PersistentClient path defaults to outputs/chroma_db (project-local)
- NPC world knowledge is loaded from personas.yaml (world_knowledge field)

Two collections per NPC in ChromaDB:
  {npc_id}_world  — static world knowledge (injected once at startup)
  {npc_id}_conv   — conversational turns (appended each turn)

At inference time, retrieve_context(query) returns the top-k semantically
relevant facts + past turns as a formatted string that is injected into the
system prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions


_DEFAULT_DB_DIR = Path(__file__).parent.parent / "outputs" / "chroma_db"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ModularMemory:
    """
    ChromaDB-backed two-tier memory for a single NPC.

    Parameters
    ----------
    npc_id : str
        Unique NPC identifier (e.g. "innkeeper_marta"). Used as collection prefix.
    db_dir : Path | str | None
        Where to persist the ChromaDB data. Defaults to outputs/chroma_db.
    embedding_model : str
        SentenceTransformer model name for embeddings. Default: all-MiniLM-L6-v2.
        (Same model already used by the guard classifier — no extra download.)
    """

    def __init__(
        self,
        npc_id: str,
        db_dir: Path | str | None = None,
        embedding_model: str = _EMBEDDING_MODEL,
    ):
        self.npc_id = npc_id
        db_path = str(db_dir or _DEFAULT_DB_DIR)
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self._world  = self._client.get_or_create_collection(
            name=f"{npc_id}_world", embedding_function=self._ef
        )
        self._conv   = self._client.get_or_create_collection(
            name=f"{npc_id}_conv",  embedding_function=self._ef
        )
        self._turn_counter = self._conv.count()  # resume from existing turns

    # ── World Knowledge ───────────────────────────────────────────────────────

    def inject_world_knowledge(self, knowledge_list: List[str]) -> None:
        """
        (Re-)inject static world knowledge into the world collection.
        Called once at CLI startup. Wipes and replaces any existing entries
        so changes in personas.yaml take effect on next run.
        """
        if self._world.count() > 0:
            existing_ids = self._world.get()["ids"]
            self._world.delete(ids=existing_ids)

        if not knowledge_list:
            return

        ids = [f"wk_{i}" for i in range(len(knowledge_list))]
        self._world.add(documents=knowledge_list, ids=ids)

    # ── Conversational Memory ─────────────────────────────────────────────────

    def add_turn(self, player_msg: str, npc_msg: str) -> None:
        """Persist one dialogue turn to the conversational memory collection."""
        doc = f"Player: {player_msg}\nNPC: {npc_msg}"
        turn_id = f"turn_{self._turn_counter}"
        self._conv.add(documents=[doc], ids=[turn_id])
        self._turn_counter += 1

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        n_world: int = 3,
        n_conv: int = 2,
    ) -> str:
        """
        Retrieve the most relevant world facts and past conversation turns
        for the given query.

        Returns a formatted string to be injected into the NPC system prompt,
        or an empty string if nothing relevant is found.
        """
        parts: list[str] = []

        # World knowledge
        if self._world.count() > 0:
            n = min(n_world, self._world.count())
            results = self._world.query(query_texts=[query], n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                parts.append("Relevant facts:\n" + "\n".join(f"- {d}" for d in docs))

        # Conversational memory
        if self._conv.count() > 0:
            n = min(n_conv, self._conv.count())
            results = self._conv.query(query_texts=[query], n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                # Reverse so they read oldest-first (semantic search is not temporal)
                parts.append(
                    "Related past exchanges:\n" + "\n".join(reversed(docs))
                )

        return "\n\n".join(parts)
