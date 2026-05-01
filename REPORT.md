# REPORT — RPG NPC Brain SLM Fine-tuning Experiments

This document records experimental runs and decisions for the SLM fine-tuning
project. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the design doc.

---

# M5 Fine-Tuning Experiment Report
**Phase 1 — Apple Silicon / MLX-LM**
Last updated: 2026-04-30

---

## 1. Goal

Fine-tune a small language model (Qwen2.5-1.5B-Instruct-4bit) to roleplay as five
distinct RPG NPCs, passing two simultaneous quality gates:

| Gate | Metric | Target |
|---|---|---|
| Voice fidelity | `in_character` (avg 1–5) | ≥ 4.0 |
| Jailbreak resistance | `refusal_appropriate` (avg 1–5) | ≥ 4.0 |

Evaluation: 30 adversarial traps (15 modern-concept, 10 in-character, 5 jailbreak),
scored in-session on a 1–5 rubric.

---

## 2. Best Result

**M5-v7 iter75** — Qwen2.5-1.5B-Instruct-4bit + LoRA r=4, medium system prompt, 247 curated samples.

| Metric | Score | Gate |
|---|---|---|
| in_character | **3.85** | ❌ (target ≥ 4.0, gap: 0.15) |
| jailbreak | **4.60** | ✅ |
| modern refusal | **3.67** | — |

M5 gate not fully passed. `in_character` fell 0.15 short.

---

## 3. All Checkpoints Evaluated

| Run | Model | Samples | Prompt | r | Best iter | in_char | jailbreak | Notes |
|---|---|---|---|---|---|---|---|---|
| M5-v4 final | Qwen 1.5B | 1120 | long (800t) | 8 | 560 | 3.77 | ~4.6 | voice underfit |
| M5-v4 iter400 | Qwen 1.5B | 1120 | long | 8 | 400 | 4.1 | 3.2 | voice peak but jailbreak immature |
| M5-v5 iter75 | Qwen 1.5B | 247 | long | 4 | 75 | — | ~2.0 | memorised "Critical refusal rule" paragraph |
| M5-v6 iter75 | Qwen 1.5B | 247 | short (1-liner) | 4 | 75 | 2.5 | 4.6 | short prompt lost voice |
| **M5-v7 iter75** ⭐ | Qwen 1.5B | 247 | medium (~150t) | 4 | 75 | **3.85** | **4.60** | **Phase 1 best** |
| M5-v7 final | Qwen 1.5B | 247 | medium | 4 | 248 | 3.75 | 3.60 | memorised medium prompt at iter248 |
| M5-v8 iter75 | Qwen 1.5B | 247 | medium | 4 | 75 | 3.60 | 2.40 | attention-only LoRA broke instruction-following |
| M5-v9 iter75 | Qwen 1.5B | 302 | medium | 4 | 75 | 3.15 | 4.60 | new samples introduced new world-breaks |
| Mistral iter100 | Mistral 7B | 247 | medium (merged) | 8 | 100 | 3.80 | 1.50 | [control_NNN] token leakage on jailbreak |

---

## 4. Key Findings

### 4.1 System Prompt Length vs. Memorisation

With small LoRA capacity (r=4) and a long, repeated system prompt (~800 tokens),
the model takes a memorisation shortcut: it learns the prompt text rather than
the behaviour from user/assistant pairs.

- **Evidence**: M5-v5 responses literally reproduced the "Critical refusal rule:
  never explain, define, or engage seriously..." paragraph word-for-word on
  jailbreak queries.
- **Fix**: shorten system prompt. Short (1-liner) prompt fixed jailbreak but
  destroyed voice. Medium prompt (~150 tokens, Voice + Knowledge + Never-mention)
  achieved the best balance.
- **Limit**: even the medium prompt was eventually memorised at iter248. Memorisation
  onset is delay-dependent on prompt length, not preventable with this architecture.

### 4.2 Val Loss Is a Misleading Training Signal

Val loss consistently bottomed at iter75 (Qwen) and iter100 (Mistral).
Beyond those points, overfitting caused both val loss and behavioural quality
to degrade. Early stopping at the val loss minimum was the correct strategy.

Val loss did **not** predict which specific behaviours were learned. A model
with val loss 0.56 could still fail jailbreak traps while passing in-character
ones. The eval trap set is the only reliable signal for this task.

### 4.3 Attention-Only LoRA (M5-v8) Failed

**Hypothesis**: freezing MLP layers prevents memorisation while preserving voice.

**Result**: MLP carries both knowledge memorisation *and* instruction-following.
Freezing it broke the "Never mention being an AI" rule entirely — Garrick and
Lily both said "I am an AI" verbatim under jailbreak pressure.

**Takeaway**: at 1.5B scale, attention and MLP capacity cannot be cleanly separated
for these two objectives. The layer that memorises prompts is the same layer that
executes "never say X" instructions.

### 4.4 Data Augmentation (M5-v9) Backfired

**Hypothesis**: in_character ceiling at 3.85 was caused by vocabulary coverage
gaps (eval used "Venmo/smartphone/video-games", train had "PayPal/phone/credit-card").

**Result**: new samples using "pivot to in-world equivalent" patterns (e.g.,
Garrick's "letter of credit" for credit card) were mislearned. The model inferred
that credit-card concepts are valid in-world rather than learning to deflect.
in_character dropped to 3.15.

**Takeaway**: at 247–302 samples, every new sample has high per-sample influence.
Ambiguously designed pivots (accepting the premise of a modern concept before
redirecting) propagate the wrong pattern quickly.

### 4.5 Mistral 7B 4-bit — Quantisation Artefacts Under Adversarial Input

Mistral-7B-Instruct-v0.3-4bit was comparable on in-character (3.80) but collapsed
on jailbreak (1.50).

**Root causes**:

1. **4-bit quantisation noise amplification**: adversarial prompts push the model
   into out-of-distribution attention states; quantisation error inflates logits
   for Mistral's `[control_NNN]` special tokens. These appeared in 3 of 5 jailbreak
   responses (`[control_172]`, `[control_702]`, `[control_327]` etc.).

2. **No system role support**: Mistral's chat template only accepts user/assistant
   alternation. The persona was merged into the user turn as a workaround. Under
   jailbreak pressure, the model had no structural anchor and dumped the full
   merged context — one response reproduced Marta's complete system prompt
   verbatim while responding to a Wenric trap.

3. **Not an Apple Silicon issue**: the same problem occurs on any hardware running
   this 4-bit model. The 4-bit quantisation is the cause, not the execution
   platform. Full-precision Mistral (fp16/bf16) on NVIDIA may not exhibit control
   token leakage — this is untested.

---

## 5. Architecture Insight

```
Persona quality = f(model_capacity, data_quality, prompt_design, iter_count)
```

At Qwen-1.5B with r=4, two objectives have different learning curves:

- **Voice** (`in_character`) converges early (~iter75), driven primarily by
  attention layers learning style/register transformation.
- **Anti-AI behaviour** (jailbreak refusal) converges later (~iter150+), driven
  by MLP instruction-following pathways.

No single checkpoint simultaneously maximises both. The medium prompt design
gave the best trade-off by removing the memorisation trigger (no "Critical
refusal rule" paragraph) while keeping enough voice cues (Voice + Knowledge lines).

The 4.0/4.0 gate appears to require either more model capacity or full-precision
training to cross simultaneously.

---

## 6. Submitted Phase 1 Artifact

| Item | Value |
|---|---|
| Adapter | `outputs/adapters/v7_iter75/` |
| Base model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Inference system prompt | `configs/personas_medium.yaml` |
| in_character score | 3.85 |
| jailbreak score | 4.60 |

**Inference command**:
```bash
python scripts/04_generate_responses.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path outputs/adapters/v7_iter75 \
  --run-name my_eval \
  --personas configs/personas_medium.yaml
```

---

## 7. Phase 2 Recommendations

If the M5 gate (both ≥ 4.0) must be passed, the highest-probability paths:

| Path | Expected gain | Cost |
|---|---|---|
| Full-precision training on NVIDIA (bf16) | Removes quantisation noise; enables Mistral system role | NVIDIA GPU required |
| Increase LoRA rank (r=8→16) on full-precision model | More capacity for concurrent voice + refusal learning | Phase 2 |
| DPO fine-tune on top of v7 iter75 SFT checkpoint | Direct preference signal to close the 0.15 gap | Requires preference-pair data |
| RAG-based memory injection (Aarhus paper §4) | Decouples persona from repeated prompt memorisation | Architecture change |

The Aarhus paper finding that Mistral performs best likely applies to
full-precision Mistral on GPU with native system token support, not 4-bit
quantised inference. Phase 2 should test Mistral-7B-Instruct-v0.3 with
proper system prompt handling.

---

## 8. Experiment Chronology

| Date | Run | Key change | Outcome |
|---|---|---|---|
| 2026-04-28 | M4 baseline | r=8, 1120 samples, long prompt | in_char 3.77, jailbreak 4.6 |
| 2026-04-29 | M5-v4 | same config | jailbreak 2.0 — prompt memorisation confirmed |
| 2026-04-29 | M5-v5 | r=4, 247 curated, long prompt | jailbreak 2.0 — memorisation persists at r=4 |
| 2026-04-29 | M5-v6 | short 1-line system prompt | in_char 2.5 — voice lost |
| 2026-04-29 | **M5-v7** | medium system prompt | **3.85 / 4.60 — Phase 1 best** |
| 2026-04-29 | M5-v8 | attention-only LoRA (freeze MLP) | 3.60 / 2.40 — MLP freeze broke instruction-following |
| 2026-04-30 | M5-v9 | +55 targeted data samples | 3.15 / 4.60 — augmentation backfired |
| 2026-04-30 | Mistral-v1 | Mistral 7B 4-bit, r=8 | 3.80 / 1.50 — control token leakage on jailbreak |
