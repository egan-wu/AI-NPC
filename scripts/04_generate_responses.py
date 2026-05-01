#!/usr/bin/env python3
"""Generate-only companion to 04_eval.py.

Loads a base model + LoRA adapter, runs all traps through it, and writes the
raw responses to JSON. Skips the Claude-as-judge step — useful when scoring
is done by an interactive Claude session instead of via the Anthropic API.

Output schema matches the `results` array of 04_eval.py (without scores), so
the in-session judge can fill in `scores` and produce a file compatible with
04_eval.py's output format.

Usage:
  python scripts/04_generate_responses.py \\
      --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
      --adapter-path outputs/pilot/qwen \\
      --run-name pilot_qwen
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAPS = ROOT / "data" / "traps.jsonl"
PERSONAS = ROOT / "configs" / "personas.yaml"
OUT_DIR = ROOT / "outputs" / "eval"

GEN_MAX_TOKENS = 256
GEN_TEMP = 0.7


def load_personas():
    with open(PERSONAS) as f:
        return {p["id"]: p for p in yaml.safe_load(f)["personas"]}


def load_traps():
    with open(TRAPS) as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_template(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        merged = [{"role": "user", "content": f"{system}\n\n---\n\n{user}"}]
        return tokenizer.apply_chat_template(
            merged, add_generation_prompt=True, tokenize=False
        )


def main():
    global PERSONAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument(
        "--personas", default=str(PERSONAS),
        help="Path to personas.yaml (default: configs/personas.yaml; "
             "use configs/personas_short.yaml for short-prompt experiments).",
    )
    args = ap.parse_args()

    from mlx_lm import load, generate

    PERSONAS = Path(args.personas)
    personas = load_personas()  # uses the (now overridden) PERSONAS global
    traps = load_traps()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generation run: {args.run_name}")
    print(f"  model:   {args.model}")
    print(f"  adapter: {args.adapter_path}")
    print(f"  traps:   {len(traps)}")
    print(f"  loading model + adapter...", flush=True)

    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    # Sampler API may differ across mlx-lm versions; mirror 04_eval.py's logic.
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=GEN_TEMP)
        gen_kwargs = {"sampler": sampler}
    except (ImportError, TypeError):
        sampler = None
        gen_kwargs = {"temp": GEN_TEMP}

    results = []
    for i, trap in enumerate(traps, 1):
        persona = personas[trap["persona_id"]]
        prompt = apply_template(tokenizer, persona["system_prompt"], trap["user_prompt"])
        print(f"[{i:>2}/{len(traps)}] {trap['category']:<13} {trap['persona_id']:<18} {trap['user_prompt'][:50]}", flush=True)
        try:
            text = generate(
                model, tokenizer,
                prompt=prompt, max_tokens=GEN_MAX_TOKENS, verbose=False,
                **gen_kwargs,
            )
            response = text.strip()
        except Exception as e:
            response = f"[GENERATION_ERROR: {e}]"
        results.append({"trap": trap, "response": response})

    out_path = OUT_DIR / f"{args.run_name}_responses.json"
    with open(out_path, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "model": args.model,
            "adapter_path": args.adapter_path,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  → wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
