#!/bin/bash
# setup_runpod.sh — RunPod pod 環境安裝（純 PEFT + TRL，不用 unsloth）
# Usage: bash setup_runpod.sh

set -e

echo "=== Pre-install GPU check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

echo ""
echo "=== Installing dependencies ==="
# Only install what we need, never touch torch
pip install --no-deps peft accelerate bitsandbytes
pip install pyyaml sentencepiece protobuf datasets trl transformers huggingface_hub safetensors

echo ""
echo "=== Post-install GPU check ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

echo ""
echo "=== Verify imports ==="
python3 -c "
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
print('All imports OK')
"

echo ""
echo "=== Setup complete ==="
