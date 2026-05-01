#!/usr/bin/env python3
"""Train (or LoRA fine-tune) via MLX-LM.

Thin wrapper around `mlx_lm.lora`. Adds:
  - Tee output to outputs/logs/<adapter_name>_<timestamp>.log
  - Pretty echo of the resolved command

All CLI args after the script name are passed through to mlx_lm.lora.

Usage:
  # M3 pilot for one candidate:
  python scripts/03_train_mlx.py \\
      --config configs/pilot.yaml \\
      --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
      --adapter-path outputs/pilot/qwen

  # M4 full training:
  python scripts/03_train_mlx.py --config configs/training.yaml
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "outputs" / "logs"


def main():
    args = sys.argv[1:]

    # Use --adapter-path value as the log name suffix
    name = "train"
    for i, a in enumerate(args):
        if a in ("--adapter-path", "-a") and i + 1 < len(args):
            name = Path(args[i + 1]).name
            break

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}_{int(time.time())}.log"

    cmd = [sys.executable, "-m", "mlx_lm.lora", *args]
    print(f"+ {' '.join(cmd)}", flush=True)
    print(f"  logging to {log_path.relative_to(ROOT)}", flush=True)

    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        proc.wait()

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
