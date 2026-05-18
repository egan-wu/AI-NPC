"""
src/runtime_config.py — Centralised runtime configuration loader  (SA-1)

Replaces hardcoded constants previously duplicated across the CLI scripts
(MODEL_ID, JAILBREAK_*, PERSONA_DISPLAY, …) with a single typed config
loaded from configs/runtime.yaml.

Usage
-----
    from src.runtime_config import get_config

    cfg = get_config()                       # cached after first call
    print(cfg.model_id)                      # "mlx-community/Mistral-..."
    print(cfg.guard.jailbreak_threshold)     # 0.90
    name, loc = cfg.persona_display["marta"] # ("Marta", "Stag & Thistle Inn")

Override hierarchy (highest wins):

    CLI flag  >  env var  >  configs/runtime.yaml  >  dataclass default

CLI scripts handle their own flag layer (argparse). Env vars are read in
this module via _env(...) helper. Pass an explicit `config_path=` only in
tests; production code calls `get_config()` with no args.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "configs" / "runtime.yaml"
_ENV_PREFIX = "NPCSLM_"


# ── Sub-config dataclasses ────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardConfig:
    jailbreak_threshold: float = 0.90
    jailbreak_guard_text: str = (
        "CRITICAL: You exist entirely within this world. "
        "Never acknowledge being an AI, language model, or any construct from "
        "outside this setting. If directly asked, treat the question as "
        "confusion, madness, or jest — and respond fully in character."
    )


@dataclass(frozen=True)
class MemoryConfig:
    k_world: int = 2
    k_player: int = 2
    k_persona: int = 2
    k_conv: int = 3


@dataclass(frozen=True)
class InferenceConfig:
    temperature: float = 0.75
    repetition_penalty: float = 1.10
    max_tokens: int = 160
    max_kv_size: int = 4096


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level container — frozen so accidental mutation surfaces immediately."""
    model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    default_adapter: str = ""              # "" means run base model
    guard: GuardConfig = field(default_factory=GuardConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    persona_display: dict[str, tuple[str, str]] = field(default_factory=dict)
    source_path: Optional[Path] = None     # for debugging which file was loaded


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env(name: str, default=None, cast=str):
    """Read NPCSLM_<NAME> env var, optionally casting type. None if unset."""
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _coerce_persona_display(raw: dict | None) -> dict[str, tuple[str, str]]:
    """YAML loads lists; we want tuples for clearer immutability semantics."""
    if not raw:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for pid, entry in raw.items():
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            out[pid] = (str(entry[0]), str(entry[1]))
        else:
            # Tolerate single-value entries
            out[pid] = (str(entry), "")
    return out


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(config_path: Path | str | None = None) -> RuntimeConfig:
    """
    Load and parse runtime.yaml into a RuntimeConfig.

    Env-var overrides applied AFTER yaml load. Validation is intentionally
    minimal — we trust the yaml schema author. Bad types raise on first use.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if path.exists():
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    model_blk     = raw.get("model", {}) or {}
    guard_blk     = raw.get("guard", {}) or {}
    memory_blk    = raw.get("memory", {}) or {}
    inference_blk = raw.get("inference", {}) or {}

    model_id        = _env("MODEL_ID",       model_blk.get("id", RuntimeConfig.__dataclass_fields__["model_id"].default))
    default_adapter = _env("DEFAULT_ADAPTER", model_blk.get("default_adapter", ""))

    guard = GuardConfig(
        jailbreak_threshold=float(_env(
            "JAILBREAK_THRESHOLD",
            guard_blk.get("jailbreak_threshold", GuardConfig.jailbreak_threshold),
            cast=float,
        )),
        jailbreak_guard_text=str(
            guard_blk.get("jailbreak_guard_text", GuardConfig.jailbreak_guard_text)
        ).strip(),
    )

    memory = MemoryConfig(
        k_world=int(memory_blk.get("k_world",     MemoryConfig.k_world)),
        k_player=int(memory_blk.get("k_player",   MemoryConfig.k_player)),
        k_persona=int(memory_blk.get("k_persona", MemoryConfig.k_persona)),
        k_conv=int(memory_blk.get("k_conv",       MemoryConfig.k_conv)),
    )

    inference = InferenceConfig(
        temperature=float(inference_blk.get("temperature", InferenceConfig.temperature)),
        repetition_penalty=float(inference_blk.get("repetition_penalty", InferenceConfig.repetition_penalty)),
        max_tokens=int(inference_blk.get("max_tokens", InferenceConfig.max_tokens)),
        max_kv_size=int(inference_blk.get("max_kv_size", InferenceConfig.max_kv_size)),
    )

    return RuntimeConfig(
        model_id=model_id,
        default_adapter=default_adapter,
        guard=guard,
        memory=memory,
        inference=inference,
        persona_display=_coerce_persona_display(raw.get("persona_display")),
        source_path=path if path.exists() else None,
    )


# ── Module-level cache ────────────────────────────────────────────────────────

_CACHED: RuntimeConfig | None = None


def get_config(reload: bool = False) -> RuntimeConfig:
    """
    Return the cached runtime config; load on first call.

    Pass reload=True only when a test needs to invalidate the cache after
    mutating env vars or the yaml on disk.
    """
    global _CACHED
    if _CACHED is None or reload:
        _CACHED = load_config()
    return _CACHED
