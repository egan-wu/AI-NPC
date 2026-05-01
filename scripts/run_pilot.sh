#!/usr/bin/env bash
# M3 pilot: train + eval each candidate base model.
# Run from project root: bash scripts/run_pilot.sh
#
# Total wall time on M-series Mac w/ 24 GB: roughly 30-60 min for all 3.
set -euo pipefail

cd "$(dirname "$0")/.."

# (name | mlx-community model path)
MODELS=(
  "qwen|mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  "gemma|mlx-community/gemma-2-2b-it-4bit"
  "phi|mlx-community/Phi-3-mini-4k-instruct-4bit"
)

echo "==== Step 0: prep MLX data dirs ===="
python scripts/prep_mlx_data.py
echo

# ---- Train ----
for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"
  model="${entry##*|}"
  echo "==== Train pilot: $name ($model) ===="
  python scripts/03_train_mlx.py \
    --config configs/pilot.yaml \
    --model "$model" \
    --adapter-path "outputs/pilot/$name"
  echo
done

# ---- Behavioral eval ----
for entry in "${MODELS[@]}"; do
  name="${entry%%|*}"
  model="${entry##*|}"
  echo "==== Behavioral eval: $name ===="
  python scripts/04_eval.py \
    --model "$model" \
    --adapter-path "outputs/pilot/$name" \
    --run-name "pilot_$name"
  echo
done

echo "==== M3 pilot complete ===="
echo "Compare:"
echo "  outputs/eval/pilot_qwen.json"
echo "  outputs/eval/pilot_gemma.json"
echo "  outputs/eval/pilot_phi.json"
echo
echo "Then update REPORT.md M3 section and pick the winner for M4."
