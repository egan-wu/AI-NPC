"""
src/memory_module.py — Modular NPC Memory (ChromaDB-backed)

Adapted from vendor/AI-NPC-Personality/npc_system/memory_module.py.
Original design: "Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue
on Consumer Hardware".

Two collections per NPC in ChromaDB:

  {npc_id}_world  — persistent world knowledge.
                    Seeded once from personas.yaml on first run (when empty).
                    Managed directly via 21_manage_world_knowledge.py.
                    Never wiped by --fresh.

  {npc_id}_conv   — conversational turn history.
                    Appended each turn during a session.
                    Cleared by --fresh (start a new conversation).

At inference time, retrieve_context(query) returns the top-k semantically
relevant facts + past turns as a formatted string injected into the system prompt.
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

        self._world = self._client.get_or_create_collection(
            name=f"{npc_id}_world", embedding_function=self._ef
        )
        self._conv = self._client.get_or_create_collection(
            name=f"{npc_id}_conv", embedding_function=self._ef
        )
        self._turn_counter = self._conv.count()  # resume from existing turns

    # ── World Knowledge ───────────────────────────────────────────────────────

    def seed_world_knowledge(self, knowledge_list: List[str]) -> bool:
        """
        Seed world knowledge from personas.yaml on first run only.
        If the _world collection is already populated, this is a no-op.

        Returns True if seeding happened, False if skipped.
        """
        if self._world.count() > 0:
            return False  # already seeded — do not overwrite

        if not knowledge_list:
            return False

        ids = [f"wk_{i}" for i in range(len(knowledge_list))]
        self._world.add(documents=knowledge_list, ids=ids)
        return True

    def world_count(self) -> int:
        return self._world.count()

    # ── Conversational Memory ─────────────────────────────────────────────────

    def add_turn(self, player_msg: str, npc_msg: str) -> None:
        """Persist one dialogue turn to the conversational memory collection."""
        doc = f"Player: {player_msg}\nNPC: {npc_msg}"
        turn_id = f"turn_{self._turn_counter}"
        self._conv.add(documents=[doc], ids=[turn_id])
        self._turn_counter += 1

    def clear_conv(self) -> int:
        """
        Wipe the conversational history collection.
        Called when --fresh is passed to the CLI.
        Returns the number of turns that were cleared.
        """
        count = self._conv.count()
        if count > 0:
            self._conv.delete(ids=self._conv.get()["ids"])
        self._turn_counter = 0
        return count

    def conv_count(self) -> int:
        return self._conv.count()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        query: str,
        n_world: int = 3,
        n_conv: int = 2,
        embedding=None,
    ) -> str:
        """
        Retrieve the most relevant world facts and past conversation turns
        for the given query.

        Parameters
        ----------
        query : str
            Raw user input (used when no pre-computed embedding is supplied).
        embedding : numpy array, shape (1, dim) or (dim,), optional
            Pre-computed query embedding from the Guard's shared embedder.
            When provided, ChromaDB uses query_embeddings instead of query_texts,
            eliminating a second encode() call per turn.

        Returns a formatted string to be injected into the NPC system prompt,
        or an empty string if nothing relevant is found.
        """
        parts: list[str] = []

        # Build ChromaDB query kwargs — prefer pre-computed embedding when available
        if embedding is not None:
            # Normalise to 1-D list for ChromaDB: (1, dim) → [dim] → [[dim]]
            vec = embedding[0] if hasattr(embedding, "ndim") and embedding.ndim == 2 \
                  else embedding
            q_kwargs: dict = {"query_embeddings": [vec.tolist()]}
        else:
            q_kwargs = {"query_texts": [query]}

        # World knowledge
        if self._world.count() > 0:
            n = min(n_world, self._world.count())
            results = self._world.query(**q_kwargs, n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                parts.append("Relevant facts:\n" + "\n".join(f"- {d}" for d in docs))

        # Conversational memory
        if self._conv.count() > 0:
            n = min(n_conv, self._conv.count())
            results = self._conv.query(**q_kwargs, n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                # Reverse so they read oldest-first (semantic search is not temporal)
                parts.append(
                    "Related past exchanges:\n" + "\n".join(reversed(docs))
                )

        return "\n\n".join(parts)

    # ── World Knowledge CRUD (used by 21_manage_world_knowledge.py) ───────────

    def world_list(self) -> list[dict]:
        """
        Return all world knowledge entries as a list of dicts:
            [{"id": "wk_0", "text": "..."}, ...]
        sorted by id.
        """
        if self._world.count() == 0:
            return []
        result = self._world.get()
        pairs = list(zip(result["ids"], result["documents"]))
        pairs.sort(key=lambda x: x[0])
        return [{"id": id_, "text": doc} for id_, doc in pairs]

    def world_add(self, text: str) -> str:
        """
        Append a new world knowledge entry. Returns the new entry's id.
        Uses a monotonically increasing numeric suffix to avoid collisions.
        """
        existing_ids = self._world.get()["ids"] if self._world.count() > 0 else []
        # Find max numeric suffix
        nums = []
        for eid in existing_ids:
            try:
                nums.append(int(eid.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
        next_n = (max(nums) + 1) if nums else 0
        new_id = f"wk_{next_n}"
        self._world.add(documents=[text], ids=[new_id])
        return new_id

    def world_update(self, entry_id: str, new_text: str) -> None:
        """Update the text of an existing world knowledge entry by id."""
        self._world.update(ids=[entry_id], documents=[new_text])

    def world_delete(self, entry_id: str) -> None:
        """Delete a world knowledge entry by id."""
        self._world.delete(ids=[entry_id])
