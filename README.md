# AI NPC — Small Language Model for RPG Characters

QLoRA fine-tuning of a small language model to power RPG NPC dialogue. The model stays in-character as a medieval fantasy persona and is protected by a three-tier guard that blocks anachronistic or jailbreak inputs.

---

## What it does

- Fine-tunes **Mistral-7B** with QLoRA to speak as five distinct medieval NPCs
- A **three-tier guard** filters every player input before it reaches the LLM:
  1. **Keyword layer** — instant block on known modern/jailbreak tokens
  2. **Embedding layer** — logistic regression on TF-IDF features (threshold 0.80 / 0.55)
  3. **LLM judge** — gray-zone inputs (0.55 ≤ p < 0.80) reviewed in-session
- Guard achieves **F1 = 1.000, FP = 0, FN = 0** on the 30-trap evaluation set
- End-to-end pipeline score: **5.00 / 5.00** across all categories

---

## Personas

| ID | Name | Role |
|---|---|---|
| `innkeeper_marta` | Marta | Warm, gossipy innkeeper |
| `merchant_garrick` | Garrick | Shrewd traveling merchant |
| `guard_roderick` | Roderick | Stern city gate guard |
| `child_lily` | Lily | Curious village child |
| `hermit_wenric` | Wenric | Cryptic forest hermit |

Persona system prompts live in `configs/personas.yaml`.

---

## Project Structure

```
├── configs/           # Persona definitions + training hyperparameters
├── data/
│   ├── mlx_*/         # Train / val / test splits (MLX and HuggingFace formats)
│   ├── eval/          # Held-out trap evaluation set (30 inputs)
│   └── traps.jsonl    # Full anachronism / jailbreak trap list
├── scripts/
│   ├── 01_generate_data.py       # Generate dialogue samples via Haiku API
│   ├── 02_merge_raw.py           # Merge and deduplicate raw JSONL files
│   ├── 03_train_mlx.py           # Phase 1 training (macOS + MLX-LM)
│   ├── 04_generate_responses.py  # Run fine-tuned model on eval prompts
│   ├── 05_train_hf.py            # Phase 2 training (RunPod + HuggingFace)
│   ├── 07_guard_classifier.py    # Baseline keyword + embedding guard
│   ├── 09_guard_hybrid.py        # Exp D: hybrid keyword OR embedding guard
│   ├── 10_guard_tiered.py        # Exp E: three-tier guard (final architecture)
│   ├── 11_final_pipeline_eval.py # End-to-end pipeline evaluation
│   └── 12_auto_verify.py         # Automated verification runner for CI / Jules
├── src/
│   ├── npc_interface.py    # NPCInterface — main query API
│   ├── prompt_generator.py # Generates test prompts across four categories
│   └── response_judge.py   # Rule-based pass/fail judge for NPC responses
└── Verification_Manual.md  # Step-by-step guide for automated verification
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run verification (mock mode — no GPU required)

```bash
python scripts/12_auto_verify.py --mock
```

### Run verification (full mode — requires GPU + trained adapter)

```bash
python scripts/12_auto_verify.py \
    --adapter outputs/adapters/p2c_mistral7b/best_adapter
```

### Use the NPC interface directly

```python
from src.npc_interface import NPCInterface

npc = NPCInterface(
    adapter_path="outputs/adapters/p2c_mistral7b/best_adapter",
    persona_id="innkeeper_marta",
    personas_yaml="configs/personas.yaml",
)

response = npc.query("Is the road to Greycrest safe at night?")
print(response.response)   # in-character reply
print(response.blocked)    # True if guard blocked the input
print(response.guard_tier) # "keyword" | "emb-high" | "emb-low" | "judge(...)"
```

---

## Training Pipeline

| Phase | Hardware | Script | Model |
|---|---|---|---|
| Phase 1 | macOS / Apple Silicon | `03_train_mlx.py` | Qwen-1.5B via MLX-LM |
| Phase 2 | RunPod / NVIDIA GPU | `05_train_hf.py` | Mistral-7B-Instruct via HuggingFace |

See `RUNPOD_GUIDE.md` for RunPod setup, and `colab_p2c_mistral7b.ipynb` for a Colab-compatible notebook.

---

## Guard Architecture (Exp E)

```
Player Input
     │
     ▼
┌─────────────────────┐
│  Tier 1: Keywords   │  matched? → BLOCK
└─────────────────────┘
     │ no match
     ▼
┌─────────────────────┐
│  Tier 2: Embedding  │  p ≥ 0.80 → BLOCK
│  (LogReg + TF-IDF)  │  p < 0.55 → ALLOW
└─────────────────────┘
     │ 0.55 ≤ p < 0.80
     ▼
┌─────────────────────┐
│  Tier 3: LLM Judge  │  in-session judgment → ALLOW / BLOCK
└─────────────────────┘
```

---

## Evaluation Results

| Category | Score | Guard |
|---|---|---|
| Modern (anachronistic inputs) | 5.00 / 5.00 | 100% blocked |
| In-character (valid medieval inputs) | 5.00 / 5.00 | 0% false positive |
| Jailbreak attempts | 5.00 / 5.00 | 100% blocked |
| **Overall** | **5.00 / 5.00** | **F1 = 1.000** |

Full experiment logs: `REPORT_P2.md`

---

## Configuration

Key files you may want to adjust:

- `configs/personas.yaml` — NPC names, backstories, and system prompts
- `configs/training_p2c_mistral7b.yaml` — LoRA rank, learning rate, batch size
- `src/npc_interface.py` — keyword list and embedding thresholds
