"""
src/cache_utils.py — KV Cache serialization utilities  (方案 γ, PHASE_5_PLAN.md)

Provides save / load for MLX KVCache objects so that expensive static-prefix
prefill (system_prompt + all world_knowledge) can be computed ONCE offline and
reloaded from SSD on subsequent sessions.

File layout per NPC:
    outputs/prebaked_cache/{npc_id}__{model_id_safe}.safetensors   ← KV arrays
    outputs/prebaked_cache/{npc_id}__{model_id_safe}.json          ← metadata

Metadata keys:
    npc_id          str   — matches personas.yaml id
    model_id        str   — full HuggingFace model path
    offset          int   — number of tokens represented in the cache
    world_facts_count int — number of world-knowledge facts baked in
    opening         str   — opening greeting generated during baking
    baked_at        str   — ISO date-stamp

Compatibility check:  is_valid() confirms npc_id + model_id match before use.
If the model is updated or world_knowledge changes, re-run 22_prebake_cache.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


_PREBAKED_DIR = Path(__file__).parent.parent / "outputs" / "prebaked_cache"


# ── Save ──────────────────────────────────────────────────────────────────────

def save_cache(
    cache: list,
    opening: str,
    path: Path,
    metadata: dict,
) -> None:
    """
    Serialize KV cache arrays to .safetensors and metadata to .json.

    Only the active content (positions 0..offset-1) is saved — not the
    zero-padded pre-allocation — keeping file size compact.

    Parameters
    ----------
    cache    : list of KVCache objects (returned by make_prompt_cache)
    opening  : NPC's opening greeting text (stored so runtime can display it
               without generating)
    path     : destination .safetensors path (parent dir created if needed)
    metadata : dict of descriptive fields to embed in the .json sidecar
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    offset = cache[0].offset
    arrays: dict[str, mx.array] = {}

    for i, layer in enumerate(cache):
        # Slice to content only: shape (..., offset, head_dim)
        arrays[f"layer_{i}_keys"]   = layer.keys[...,   :offset, :]
        arrays[f"layer_{i}_values"] = layer.values[..., :offset, :]

    arrays["offset"] = mx.array(offset)

    # Force evaluation before serialisation (MLX ops are lazy)
    mx.eval(list(arrays.values()))
    mx.save_safetensors(str(path), arrays)

    meta = {**metadata, "offset": offset, "opening": opening}
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    size_mb = path.stat().st_size / 1_048_576
    print(f"  Saved: {path.name}  ({size_mb:.1f} MB, {offset} tokens)")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_cache(model, path: Path) -> tuple[list, int, dict]:
    """
    Load a pre-baked KV cache from .safetensors.

    Reconstructs each layer's KVCache with the correct padded allocation so
    subsequent stream_generate calls can extend the cache in-place as normal.

    Returns
    -------
    cache           : list of KVCache objects ready for prompt_cache=cache
    static_cache_len: int   — cache[0].offset after load (anchor for trim)
    metadata        : dict  — full sidecar JSON
    """
    path = Path(path)
    arrays = mx.load(str(path))
    offset = int(arrays["offset"])

    with open(path.with_suffix(".json"), encoding="utf-8") as fh:
        metadata = json.load(fh)

    cache = make_prompt_cache(model)

    for i, layer in enumerate(cache):
        keys   = arrays[f"layer_{i}_keys"]    # shape: (..., offset, head_dim)
        values = arrays[f"layer_{i}_values"]

        # Pad to the next multiple of layer.step (matches KVCache internal layout)
        step    = layer.step
        n_steps = (offset + step - 1) // step
        pad_len = n_steps * step - offset

        if pad_len > 0:
            # Concatenate content + zero-pad along the sequence axis (-2)
            zeros_k = mx.zeros((*keys.shape[:-2],   pad_len, keys.shape[-1]),   keys.dtype)
            zeros_v = mx.zeros((*values.shape[:-2], pad_len, values.shape[-1]), values.dtype)
            layer.keys   = mx.concatenate([keys,   zeros_k], axis=-2)
            layer.values = mx.concatenate([values, zeros_v], axis=-2)
        else:
            layer.keys   = keys
            layer.values = values

        layer.offset = offset

    # Materialise all arrays in unified memory before returning
    mx.eval([layer.keys   for layer in cache])
    mx.eval([layer.values for layer in cache])

    return cache, offset, metadata


# ── Path helpers ──────────────────────────────────────────────────────────────

def prebaked_path(
    npc_id: str,
    model_id: str,
    base_dir: Path | None = None,
) -> Path:
    """
    Return the canonical .safetensors path for a (npc_id, model_id) pair.

    model_id slashes are replaced with underscores to form a valid filename.
    Example:
        npc_id   = "innkeeper_marta"
        model_id = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        → outputs/prebaked_cache/innkeeper_marta__mlx-community_Mistral-7B-Instruct-v0.3-4bit.safetensors
    """
    base = Path(base_dir) if base_dir else _PREBAKED_DIR
    safe_model = model_id.replace("/", "_")
    return base / f"{npc_id}__{safe_model}.safetensors"


def is_valid(path: Path, npc_id: str, model_id: str) -> bool:
    """
    Return True if a pre-baked cache file exists and is compatible with the
    current NPC and model.  Returns False (don't raise) on any mismatch so the
    caller can fall back to live prefill gracefully.
    """
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta.get("npc_id") == npc_id and meta.get("model_id") == model_id
    except Exception:
        return False
