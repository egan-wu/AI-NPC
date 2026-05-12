"""
src/memory_module.py — Modular NPC Memory (ChromaDB-backed)

Adapted from vendor/AI-NPC-Personality/npc_system/memory_module.py.
Original design: "Fixed-Persona SLMs with Modular Memory: Scalable NPC Dialogue
on Consumer Hardware".

Two collections per NPC in ChromaDB:

  {npc_id}_persona — persistent personal background facts (L3 of the
                     hierarchy). Seeded once from personas.yaml on first run.
                     Managed via 21_manage_persona_lore.py. Never wiped by
                     --fresh. Renamed from `_world` in Phase 6 to disambiguate
                     from the new shared `world_global` collection (L0).

  {npc_id}_conv    — conversational turn history (L4). Appended each turn.
                     Cleared by --fresh.

At inference time, retrieve_context(query) returns the top-k semantically
relevant persona facts + past turns. For full-hierarchy retrieval (L0 + L_p
+ L3 + L4), use src/memory_hierarchy.HierarchicalMemory.
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

        self._persona = self._client.get_or_create_collection(
            name=f"{npc_id}_persona", embedding_function=self._ef
        )
        self._conv = self._client.get_or_create_collection(
            name=f"{npc_id}_conv", embedding_function=self._ef
        )
        self._turn_counter = self._conv.count()  # resume from existing turns

    # ── Persona Lore (L3) ─────────────────────────────────────────────────────

    def seed_persona_lore(self, knowledge_list: List[str]) -> bool:
        """
        Seed persona lore from personas.yaml on first run only.
        If the _persona collection is already populated, this is a no-op.

        Returns True if seeding happened, False if skipped.
        """
        if self._persona.count() > 0:
            return False  # already seeded — do not overwrite

        if not knowledge_list:
            return False

        ids = [f"pl_{i}" for i in range(len(knowledge_list))]
        self._persona.add(documents=knowledge_list, ids=ids)
        return True

    def persona_count(self) -> int:
        return self._persona.count()

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
        n_persona: int = 3,
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

        # Persona lore
        if n_persona > 0 and self._persona.count() > 0:
            n = min(n_persona, self._persona.count())
            results = self._persona.query(**q_kwargs, n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                parts.append("[About me]\n" + "\n".join(f"- {d}" for d in docs))

        # Conversational memory
        if n_conv > 0 and self._conv.count() > 0:
            n = min(n_conv, self._conv.count())
            results = self._conv.query(**q_kwargs, n_results=n)
            docs = results["documents"][0] if results["documents"] else []
            if docs:
                # Reverse so they read oldest-first (semantic search is not temporal)
                parts.append(
                    "[Recent conversation]\n" + "\n".join(reversed(docs))
                )

        return "\n\n".join(parts)

    # ── Persona Lore CRUD (used by 21_manage_persona_lore.py) ────────────────

    def persona_list(self) -> list[dict]:
        """Return all persona lore entries sorted by id: [{'id', 'text'}, ...]."""
        if self._persona.count() == 0:
            return []
        result = self._persona.get()
        pairs = list(zip(result["ids"], result["documents"]))
        pairs.sort(key=lambda x: x[0])
        return [{"id": id_, "text": doc} for id_, doc in pairs]

    def persona_add(self, text: str) -> str:
        """Append a new persona lore entry. Returns the new id (pl_<n>)."""
        existing_ids = self._persona.get()["ids"] if self._persona.count() > 0 else []
        nums = []
        for eid in existing_ids:
            try:
                nums.append(int(eid.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
        next_n = (max(nums) + 1) if nums else 0
        new_id = f"pl_{next_n}"
        self._persona.add(documents=[text], ids=[new_id])
        return new_id

    def persona_update(self, entry_id: str, new_text: str) -> None:
        self._persona.update(ids=[entry_id], documents=[new_text])

    def persona_delete(self, entry_id: str) -> None:
        self._persona.delete(ids=[entry_id])

    def persona_all_texts(self) -> list[str]:
        """Return persona lore entries as plain text list, in id order."""
        return [e["text"] for e in self.persona_list()]
