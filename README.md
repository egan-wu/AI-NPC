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
- **Multi-layer memory** (Phase 6) — every NPC retrieves from shared world
  lore + the player's deeds + their own background + the running conversation
- **KV-cache optimisations** (Phase 5) — incremental β cache + offline γ
  pre-baked cache reduce TTFT from **~5.0 s → ~0.7 s** per turn

---

## Memory Hierarchy (Phase 6)

```
L0  world_global       all NPCs, immutable    configs/world_knowledge.yaml
L_p player_lore        all NPCs, runtime      24_manage_player_lore.py
L3  {npc_id}_persona   single NPC, static     personas.yaml + 21_manage_persona_lore.py
L4  {npc_id}_conv      single NPC, session    auto in 20_npc_cli_memory.py
```

Each layer is queried independently per turn (defaults: `k_world=2`,
`k_player=2`, `k_persona=2`, `k_conv=3`) and labelled in the prompt:

```
[World — common knowledge]      ← L0
[About the traveller]           ← L_p
[About me]                      ← L3
[Recent conversation]           ← L4
```

L0 + L3 are static and **bake into the γ pre-baked KV cache**;
L_p + L4 are dynamic and ride in the per-turn β delta.
See `PROJECT_PLAN.md §13` for the full spec.

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
├── configs/
│   ├── personas.yaml          # NPC backstories + system prompts + L3 lore
│   ├── world_knowledge.yaml   # L0 shared world facts (Phase 6)
│   └── training_*.yaml        # LoRA hyperparameters
├── data/
│   ├── raw/                   # Per-persona JSONL batches (v1…v9)
│   ├── processed/             # train_curated.jsonl + eval split
│   └── traps.jsonl            # Anachronism / jailbreak trap list
├── scripts/
│   ├── 01_generate_data.py        # Generate dialogue samples via Haiku API
│   ├── 02_merge_raw.py            # Merge raw JSONL files
│   ├── 02b_rebuild_curated.py     # Deterministic dedup + ranking → train_curated
│   ├── 03_train_mlx.py            # Phase 1 training (MLX-LM)
│   ├── 04_generate_responses.py   # Run fine-tuned model on eval prompts
│   ├── 05_train_hf.py             # Phase 2 training (HuggingFace)
│   ├── 10_guard_tiered.py         # Three-tier guard (final architecture)
│   ├── 11_final_pipeline_eval.py  # End-to-end pipeline evaluation
│   ├── 12_auto_verify.py          # Automated verification runner
│   ├── 19_npc_cli.py              # Basic interactive NPC CLI
│   ├── 20_npc_cli_memory.py       # CLI w/ memory hierarchy + KV cache (β + γ)
│   ├── 21_manage_persona_lore.py  # Edit per-NPC L3 background facts
│   ├── 22_prebake_cache.py        # Offline γ KV-cache baking
│   ├── 23_seed_world_knowledge.py # Sync L0 from YAML → ChromaDB
│   └── 24_manage_player_lore.py   # CRUD for L_p (player deeds, all NPCs)
├── src/
│   ├── npc_interface.py       # NPCInterface — main query API
│   ├── memory_module.py       # Per-NPC ChromaDB memory (L3 + L4)
│   ├── memory_hierarchy.py    # Phase 6 — 4-layer hierarchical memory
│   ├── cache_utils.py         # Phase 5 γ — KV cache serialisation
│   ├── prompt_generator.py    # Test prompt generator
│   └── response_judge.py      # Rule-based pass/fail judge
├── PROJECT_PLAN.md            # Full design doc (Phases 1–6)
├── PHASE_5_PLAN.md            # KV cache deep-dive
└── Verification_Manual.md     # Verification guide
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

### Interactive CLI with memory + KV cache

```bash
# 1. Seed L0 shared world facts into ChromaDB (once, after editing YAML)
python scripts/23_seed_world_knowledge.py

# 2. Pre-bake γ KV cache for fast cold starts (once per persona)
python scripts/22_prebake_cache.py --all

# 3. Chat — auto-loads γ cache + uses 4-layer hierarchical memory
python scripts/20_npc_cli_memory.py -p marta --timing

# Add a player deed visible to every NPC (next turn after restart)
python scripts/24_manage_player_lore.py add "Slew the Greycrest dragon"

# Edit a single NPC's background facts (interactive)
python scripts/21_manage_persona_lore.py -p marta
```

Useful flags on `20_npc_cli_memory.py`:
`--fresh` clear conversation, `--no-history` disable in-context history,
`--k-world / --k-player / --k-persona / --k-conv` tune per-layer retrieval,
`--prebaked-cache off` skip γ and use β live prefill, `--timing` per-stage latency.

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
