"""
src/logging_config.py — Structured logging (SA-9)

Single source of logging configuration for the whole repo. Two output sinks:

  1. stdout — human-friendly + ANSI colours by level (DEV)
  2. JSONL  — append-only at outputs/logs/app.jsonl       (PROD / debug)

Level controlled by the NPCSLM_LOG_LEVEL env var (default INFO). Set to DEBUG
when debugging cache invalidation, retrieval surprises, KV trims, etc.

Scope policy
------------
This module is for *program* events (cache hits, KV trims, ChromaDB errors,
guard verdicts) — NOT for the chat UI. The CLI's persona dialogue, settings
banner, and prompt rendering remain plain `print()` because that's UX, not
diagnostics.

Usage
-----
    from src.logging_config import get_logger
    log = get_logger(__name__)

    log.info("Cache hit", extra={"adapter_id": adapter_id, "npc": npc_id})
    log.warning("KV size %d exceeded cap %d — trimming", offset, cap)
    log.error("ChromaDB query failed: %s", err)

Each call writes a single line to stdout (formatted with colour) AND a
single JSONL row to outputs/logs/app.jsonl (with `extra=` merged into the
record). Use the `extra=` dict for structured fields; production tooling
will key off those, not free-form message strings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "outputs" / "logs"
_LOG_FILE = _LOG_DIR / "app.jsonl"

_LEVEL_ENV = "NPCSLM_LOG_LEVEL"

# ANSI colours by level
_LEVEL_COLOR = {
    "DEBUG":    "\033[2m",      # dim
    "INFO":     "\033[96m",     # cyan
    "WARNING":  "\033[93m",     # yellow
    "ERROR":    "\033[91m",     # red
    "CRITICAL": "\033[1;91m",   # bold red
}
_RESET = "\033[0m"

# Reserved attribute names on LogRecord we must NOT treat as user-provided extras
_RESERVED_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


# ── Formatters ────────────────────────────────────────────────────────────────

class _PrettyFormatter(logging.Formatter):
    """Coloured single-line format for stdout."""
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLOR.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = record.getMessage()
        extras = _extras_dict(record)
        suffix = ""
        if extras:
            suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return f"{color}{ts} {record.levelname:<5} {record.name}{_RESET}  {msg}{suffix}"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line — keys: ts, level, logger, msg, extras..."""
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extras_dict(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _extras_dict(record: logging.LogRecord) -> dict[str, Any]:
    """Pull only user-provided `extra={}` kwargs from a LogRecord."""
    return {
        k: v for k, v in record.__dict__.items()
        if k not in _RESERVED_RECORD_ATTRS and not k.startswith("_")
    }


# ── Initialiser ───────────────────────────────────────────────────────────────

_CONFIGURED = False


def _configure_root() -> None:
    """Configure root logger once. Subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get(_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("npcslm")    # repo-scoped root, not THE root
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False                # don't bubble to python's root

    # stdout (pretty)
    sh = logging.StreamHandler(stream=sys.stderr)  # stderr keeps chat UX clean
    sh.setLevel(level)
    sh.setFormatter(_PrettyFormatter())
    root.addHandler(sh)

    # JSONL file (always DEBUG to keep the audit trail comprehensive)
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)
    except OSError as e:
        # Disk full / permission — don't crash the app; warn to stderr.
        print(f"WARN: could not open log file {_LOG_FILE}: {e}", file=sys.stderr)

    _CONFIGURED = True


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger under the `npcslm.` namespace.

    Typical pattern in modules:
        from src.logging_config import get_logger
        log = get_logger(__name__)
        log.info("Did the thing", extra={"key": "value"})
    """
    _configure_root()
    # Normalise: callers pass __name__ which is "src.memory_module" etc.
    # We want them all under "npcslm.<module>".
    short = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"npcslm.{short}")
