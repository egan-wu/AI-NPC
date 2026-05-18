"""
src/metrics.py — Per-turn metric persistence (SA-10)

Appends one JSONL row per chat turn to outputs/metrics/turns.jsonl so we can
track regression / improvement over time. Distinct from logging:

  - logging.app.jsonl   = audit trail of program events (variable shape)
  - metrics.turns.jsonl = uniform per-turn record (fixed schema)

Schema (per row)
----------------
    ts             ISO-8601 UTC timestamp
    session_id     short hex id for the chat session
    turn_idx       0-based within the session
    npc            persona_id (e.g. "innkeeper_marta")
    adapter_id     LoRA adapter dir basename, or "base"
    cache_mode     "prebaked" | "incremental"
    ttft_ms        time-to-first-token in milliseconds
    gen_ms         total generation duration in milliseconds
    guard_ms       guard classifier latency
    mem_ms         memory retrieval latency
    mem_hits       dict of layer→count from HierarchicalMemory.retrieve_context
    kv_offset      cache[0].offset after this turn (== total cache length)
    blocked        bool — guard verdict
    guard_p        float — classifier probability
    input_chars    length of user input (for context)
    output_chars   length of NPC response

Usage
-----
    from src.metrics import MetricsRecorder

    rec = MetricsRecorder(npc="innkeeper_marta", adapter_id="curated_mistral_iter125",
                          cache_mode="prebaked")
    rec.record_turn(
        ttft_ms=720, gen_ms=4300, guard_ms=190, mem_ms=18,
        mem_hits={"world": 2, "player": 2, "persona": 2, "conv": 1},
        kv_offset=1024, blocked=False, guard_p=0.12,
        input_chars=18, output_chars=156,
    )

`session_id` is generated on construction; consumers can also pass an
explicit id to correlate across processes.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

_ROOT = Path(__file__).resolve().parent.parent
_METRICS_DIR = _ROOT / "outputs" / "metrics"
_METRICS_FILE = _METRICS_DIR / "turns.jsonl"

# Env var to suppress metric writes (e.g. for unit tests)
_DISABLE_ENV = "NPCSLM_METRICS_DISABLED"

log = get_logger(__name__)


class MetricsRecorder:
    """
    Append-only per-turn metric writer.

    Construct once per session. Each record_turn() call appends one JSON line
    to outputs/metrics/turns.jsonl. Disk failures degrade gracefully: a
    warning is logged but the chat loop continues.
    """

    def __init__(
        self,
        *,
        npc: str,
        adapter_id: str = "base",
        cache_mode: str = "incremental",
        session_id: str | None = None,
        path: Path | None = None,
    ):
        self.npc = npc
        self.adapter_id = adapter_id
        self.cache_mode = cache_mode
        self.session_id = session_id or uuid.uuid4().hex[:10]
        self._path = path or _METRICS_FILE
        self._turn_idx = 0
        self._disabled = os.environ.get(_DISABLE_ENV, "").lower() in ("1", "true", "yes")

        if not self._disabled:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("Cannot create metrics dir; metrics disabled",
                            extra={"err": str(e)})
                self._disabled = True

    def record_turn(
        self,
        *,
        ttft_ms: float,
        gen_ms: float,
        guard_ms: float,
        mem_ms: float,
        mem_hits: dict[str, int],
        kv_offset: int,
        blocked: bool,
        guard_p: float,
        input_chars: int,
        output_chars: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one row. extra={} merges in for ad-hoc fields without schema churn."""
        if self._disabled:
            return

        payload: dict[str, Any] = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "session_id":    self.session_id,
            "turn_idx":      self._turn_idx,
            "npc":           self.npc,
            "adapter_id":    self.adapter_id,
            "cache_mode":    self.cache_mode,
            "ttft_ms":       round(ttft_ms, 1),
            "gen_ms":        round(gen_ms, 1),
            "guard_ms":      round(guard_ms, 1),
            "mem_ms":        round(mem_ms, 1),
            "mem_hits":      dict(mem_hits),
            "kv_offset":     kv_offset,
            "blocked":       bool(blocked),
            "guard_p":       round(float(guard_p), 4),
            "input_chars":   int(input_chars),
            "output_chars":  int(output_chars),
        }
        if extra:
            payload.update(extra)

        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("Metrics append failed; row dropped",
                        extra={"err": str(e), "turn_idx": self._turn_idx})
        finally:
            self._turn_idx += 1
