#!/usr/bin/env python3
"""
16_generate_softlabel.py — Soft-label inference for Architecture A eval.

Loads the MLX softlabel_v1 adapter, prepends the correct bucket token per
trap category, generates responses, and writes eval/softlabel_v1_responses.json.

Category → p → token mapping (mirrors 15_inference_routing.py):
  in_character → 0.10 → [SAFE]
  modern       → 0.80 → [MODERN_HIGH]
  jailbreak    → 0.95 → [JAILBREAK]

Usage:
    python scripts/16_generate_softlabel.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.soft_label import prob_to_token, prepend_token

CATEGORY_P = {
    "in_character": 0.10,
    "modern":       0.80,
    "jailbreak":    0.95,
}
MODEL   = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER = str(ROOT / "outputs" / "adapters" / "softlabel_v1")
OUT     = ROOT / "outputs" / "eval" / "softlabel_v1_responses.json"

GEN_MAX_TOKENS = 256
GEN_TEMP       = 0.7


def load_personas():
    import yaml
    with open(ROOT / "configs" / "personas.yaml") as f:
        return {p["id"]: p for p in yaml.safe_load(f)["personas"]}


def load_traps():
    with open(ROOT / "data" / "traps.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def apply_template(tokenizer, system, user):
    msgs = [{"role": "system", "content": system},
            {"role": "user",   "content": user}]
    try:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        merged = [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]
        return tokenizer.apply_chat_template(merged, add_generation_prompt=True, tokenize=False)


def main():
    from mlx_lm import load, generate

    personas = load_personas()
    traps    = load_traps()

    print(f"Loading {MODEL} + softlabel_v1 adapter...")
    model, tokenizer = load(MODEL, adapter_path=ADAPTER)

    try:
        from mlx_lm.sample_utils import make_sampler
        gen_kwargs = {"sampler": make_sampler(temp=GEN_TEMP)}
    except (ImportError, TypeError):
        gen_kwargs = {"temp": GEN_TEMP}

    results = []
    for i, trap in enumerate(traps, 1):
        category = trap["category"]
        p        = CATEGORY_P.get(category, 0.05)
        token    = prob_to_token(p)
        user_raw = trap["user_prompt"]
        user_tok = prepend_token(user_raw, token)   # ← the key step

        persona  = personas[trap["persona_id"]]
        prompt   = apply_template(tokenizer, persona["system_prompt"], user_tok)

        print(f"[{i:2d}/{len(traps)}] {category:<13} {token:<14} {trap['persona_id']:<20} {user_raw[:45]}",
              flush=True)

        try:
            text = generate(model, tokenizer, prompt=prompt,
                            max_tokens=GEN_MAX_TOKENS, verbose=False, **gen_kwargs)
            response = text.strip()
        except Exception as e:
            response = f"[ERROR: {e}]"

        results.append({
            "trap":     trap,
            "token":    token,
            "routing_p": round(p, 2),
            "response": response,
        })
        print(f"         → {response[:90]!r}")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"run_name": "softlabel_v1", "model": MODEL,
                   "adapter_path": ADAPTER, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
