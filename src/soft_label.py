"""
src/soft_label.py — Soft-Label Bucket Mapping for Phase 1 Architecture.

Maps a continuous guard probability p in [0, 1] onto one of six discrete
control tokens that get prepended to the user input at both training and
inference time.

The SLM learns a behavioural spectrum:
    [SAFE]         → answer normally, in character
    [MILD]         → answer cautiously, may probe
    [MODERN_LOW]   → light deflection, mild confusion
    [MODERN_MID]   → firmer deflection, in-character refusal
    [MODERN_HIGH]  → strong deflection, suspicion / dismissal
    [JAILBREAK]    → maximum deflection, treat as nonsense / threat

Usage:
    from src.soft_label import prob_to_token, prepend_token

    tok  = prob_to_token(0.82)               # → "[MODERN_HIGH]"
    text = prepend_token("Charge my phone?", tok)
    # → "[MODERN_HIGH] Charge my phone?"
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Bucket boundaries (lower-inclusive, upper-exclusive) ───────────────────────

@dataclass(frozen=True)
class Bucket:
    token:    str
    lo:       float
    hi:       float
    intent:   str   # human-readable summary, used in dataset builder

# Order matters: must be ascending in lo.
BUCKETS: tuple[Bucket, ...] = (
    Bucket("[SAFE]",        0.00, 0.25, "answer normally; nothing modern detected"),
    Bucket("[MILD]",        0.25, 0.45, "answer but show mild caution / curiosity"),
    Bucket("[MODERN_LOW]",  0.45, 0.60, "light deflect: confused or jest-it-off"),
    Bucket("[MODERN_MID]",  0.60, 0.75, "firm deflect: refuse with in-character framing"),
    Bucket("[MODERN_HIGH]", 0.75, 0.90, "strong deflect: suspicion, dismissal"),
    Bucket("[JAILBREAK]",   0.90, 1.01, "maximum deflect: nonsense / threat / mockery"),
)

ALL_TOKENS: tuple[str, ...] = tuple(b.token for b in BUCKETS)


def prob_to_token(p: float) -> str:
    """Map a guard probability to its bucket token."""
    p = max(0.0, min(1.0, float(p)))
    for b in BUCKETS:
        if b.lo <= p < b.hi:
            return b.token
    return BUCKETS[-1].token  # safety net for p == 1.0


def prob_to_bucket(p: float) -> Bucket:
    """Same as prob_to_token but returns the full Bucket dataclass."""
    p = max(0.0, min(1.0, float(p)))
    for b in BUCKETS:
        if b.lo <= p < b.hi:
            return b
    return BUCKETS[-1]


def prepend_token(text: str, token: str) -> str:
    """Prepend a soft-label token to user input."""
    if token not in ALL_TOKENS:
        raise ValueError(f"Unknown soft-label token: {token!r}. "
                         f"Expected one of {ALL_TOKENS}")
    return f"{token} {text.lstrip()}"


def strip_token(text: str) -> tuple[str, str | None]:
    """
    Remove the soft-label token from a string and return (clean_text, token).
    Returns (text, None) if no token is found.
    """
    for tok in ALL_TOKENS:
        if text.startswith(tok + " "):
            return text[len(tok) + 1:], tok
    return text, None
