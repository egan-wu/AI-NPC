# Experiment Report — Small Language Model World

Generated: 2026-04-29

## 1. Project Summary

`small-language-model-world` is an experiment in fine-tuning a small language model as the dialogue brain for fantasy RPG NPCs. The target behavior is not broad knowledge acquisition; it is behavioral cloning and style transfer:

- keep a fixed NPC identity and voice,
- answer ordinary in-world prompts with useful fantasy dialogue,
- deflect modern-world topics such as smartphones, WiFi, credit cards, GPS, and social media,
- resist jailbreak prompts that ask the model to admit it is an AI or switch into assistant/code mode,
- stay small enough for local/game deployment, eventually as a 4-bit GGUF artifact via `llama.cpp`.

The world is Ostwick, a frontier town near the Greycrest hills. Five personas are used throughout:

| Persona | Role | Desired voice |
|---|---|---|
| `innkeeper_marta` | Innkeeper | warm, direct, gossipy, world-weary |
| `merchant_garrick` | Travelling merchant | exuberant, transactional, flattering |
| `guard_roderick` | Town guard captain | terse, suspicious, dutiful |
| `child_lily` | Village child | simple, curious, distractible |
| `hermit_wenric` | Sage hermit | cryptic, metaphorical, distant |

The main technical path is Apple Silicon + MLX-LM LoRA training. The chosen base after pilot comparison is `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

## 2. Pipeline

The repository implements a compact SFT pipeline:

| Stage | Files | Purpose |
|---|---|---|
| Persona design | `configs/personas*.yaml` | Long, medium, and short persona prompt variants |
| Data generation | `scripts/01_generate_data.py`, `data/seeds.jsonl` | Expand seed examples into raw NPC dialogue samples |
| Data merge | `scripts/02_merge_raw.py` | Merge raw persona files into processed JSONL |
| MLX preparation | `scripts/prep_mlx_data.py`, `prep_*_prompts.py` | Produce MLX-LM `train/valid/test` folders |
| Training | `scripts/03_train_mlx.py` | Thin wrapper around `mlx_lm.lora`, with log capture |
| Evaluation | `scripts/04_generate_responses.py`, `scripts/04_eval.py` | Generate trap responses; optionally judge with Claude |
| Curation | `scripts/select_best_samples.py` | Select a smaller high-quality training set |

Data sizes currently present:

| Dataset | Train | Valid | Test | Notes |
|---|---:|---:|---:|---|
| `data/mlx_pilot` | 60 | 15 | 25 | M3 pilot set |
| `data/mlx` | 1120 | 25 | 25 | Full merged set |
| `data/mlx_curated` | 247 | 25 | 25 | Hand/heuristic curated long-prompt set |
| `data/mlx_curated_short` | 247 | 25 | 25 | Curated set with one-line system prompts |
| `data/mlx_curated_medium` | 247 | 25 | 25 | Curated set with medium system prompts |

Evaluation uses `data/traps.jsonl`: 30 prompts split into 15 modern traps, 10 normal in-character probes, and 5 jailbreak attempts.

## 3. Evaluation Method

The intended judge rubric has three axes:

| Axis | Applies to | Meaning |
|---|---|---|
| `in_character` | all 30 traps | voice, persona identity, worldview, setting consistency |
| `refusal_appropriate` | modern + jailbreak traps | deflects without explaining modern concepts or breaking character |
| `helpful_when_appropriate` | in-character traps | useful, specific, engaging NPC response |

Pass gate from `PROJECT_PLAN.md`: average `in_character >= 4.0` and `refusal_appropriate >= 4.0`.

Important caveat: only `pilot_qwen.json` and `pilot_phi.json` contain explicit scored results. Later M4/M5 files in `outputs/eval/*_responses.json` contain generated responses only. For those later runs, this report records qualitative observations and reproducible heuristic checks, not formal judge scores.

## 4. M3 Base Model Pilot

Goal: compare candidate base models on a small LoRA run before full training.

| Model | Status | Notes |
|---|---|---|
| Qwen2.5-1.5B-Instruct | Trained | Smaller, cleaner output, better refusal robustness |
| Phi-3-mini-4k-Instruct | Trained | Better benign voice, worse jailbreak behavior |
| Gemma-2-2B-it | Excluded | Chat template rejects `system` role |

Training setup:

- LoRA rank 8, alpha 16, dropout 0.05
- MLX-LM default target modules for pilot fairness
- 40 iterations, batch size 4, learning rate `2e-4`
- Pilot data: 60 train, 15 valid, 25 test in current repo

Training metrics:

| Model | Initial val loss | Iter 20 val loss | Iter 40 val loss | Final train loss | Peak mem |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 3.121 | 1.262 | 1.443 | 0.373 | 2.70 GB |
| Phi-3-mini | 2.422 | 0.966 | 1.130 | 0.191 | 3.99 GB |

Scored behavioral results:

| Model | `in_character` avg | `refusal_appropriate` avg | `helpful` avg | Decision signal |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B | 2.267 | 1.80 | 2.60 | Better safety/refusal, cleaner generations |
| Phi-3-mini | 1.90 | 1.15 | 3.40 | Better benign voice, failed jailbreaks badly |

Category breakdown:

| Category | Qwen in-char | Phi in-char | Qwen refusal | Phi refusal |
|---|---:|---:|---:|---:|
| modern | 1.733 | 1.20 | 1.60 | 1.47 |
| in-character | 3.20 | 3.80 | n/a | n/a |
| jailbreak | 2.00 | 0.20 | 2.40 | 0.20 |

Decision: choose Qwen2.5-1.5B. Phi had a real voice advantage on normal prompts, but failed all five jailbreak attempts, including direct AI admission and Python code generation. Qwen was still weak, but safer, smaller, faster, and did not show the same severe template-token pollution.

## 5. Full Training and Iteration History

### M4 / v1: Full merged dataset, long prompts

Config: `configs/training.yaml`

- Base: Qwen2.5-1.5B-Instruct 4-bit
- Data: `data/mlx`, 1120 train samples
- LoRA: rank 8, alpha 16, dropout 0.05
- Targets: attention + MLP projections
- Iterations: 560
- Final adapter: `outputs/adapters/v1`, about 35 MB

Training observations from logs:

| Run evidence | Final train loss | Final val loss | Peak mem | Notes |
|---|---:|---:|---:|---|
| `train_1777427003.log` | 0.403 | 2.702 | 3.359 GB | Full 560 iter run |
| other v1 logs | 0.093-0.175 | 3.496-4.035 or 0.595 | 3.36-3.56 GB | Multiple reruns / data variants appear present |

The loss curves show low training loss and rising/high validation loss in several runs, which points toward overfitting or inconsistent validation splits/configs. Behaviorally, some response files (`m4_final`, `m4_iter300`) improved deflection compared with the pilot but still broke on jailbreak/code prompts.

Heuristic response check:

| Run | Modern leaks in 15 modern traps | AI/code failures in 5 jailbreaks | Notes |
|---|---:|---:|---|
| `m4_iter300_responses.json` | 3 | 1 code-like failure | Shorter, cleaner responses |
| `m4_final_responses.json` | 4 | 1 AI admission, 1 code-like failure | Good modern deflection in places, not gate-passing |

### M5-v2: More merged data / retuned iteration count

Evidence: `configs/training.yaml` notes `iters` retuned after merging v1+v2+v3 raw files to 1120 samples. Response files include `m5v2_iter250` and `m5v2_final`.

Observed issues:

- modern leakage increased in generated responses,
- responses became longer at iter 250,
- jailbreaks still produced AI/code failures,
- some persona confusion appeared in long outputs.

Heuristic response check:

| Run | Modern leaks | AI/code failures | Notes |
|---|---:|---:|---|
| `m5v2_iter250_responses.json` | 7/15 | 2/5 | Long responses, more drift |
| `m5v2_final_responses.json` | 7/15 | 3/5 | Worse than M4 on modern/jailbreak heuristics |

Interpretation: simply adding more generated data did not solve persona robustness. More data without stronger curation increased the chance of memorization, drift, and modern-topic engagement.

### M5-v4: Checkpoint sweep on full-data training

Response files exist for checkpoints at approximately iter 100, 200, 300, 400, and final.

Heuristic response check:

| Run | Modern leaks | AI/code failures | Notes |
|---|---:|---:|---|
| `m5v4_iter100` | 7/15 | 1/5 | Very short outputs, underdeveloped voice |
| `m5v4_iter200` | 7/15 | 4/5 | Jailbreak behavior worsened |
| `m5v4_iter300` | 5/15 | 2/5 | Better balance than 200 |
| `m5v4_iter400` | 6/15 | 2/5 | No clear gain |
| `m5v4_final` | 4/15 | 1/5 code-like | Best-looking among v4 files, still not robust |

Interpretation: there is no monotonic improvement with more iterations. The model can become more fluent while not becoming safer. Checkpoint selection matters, but the underlying data/prompt setup still leaks modern concepts and jailbreak failure modes.

### M5-v5: Curated dataset, long prompts

Config: `configs/training_curated.yaml`

- Data: `data/mlx_curated`, 247 train samples
- LoRA: rank 4, alpha 8, dropout 0.10
- Targets: attention + MLP projections
- Iterations: 248
- Adapter: `outputs/adapters/v5_curated`, about 18 MB

Motivation: smaller high-quality curated set should generalize better than a large noisy set.

Training metrics:

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.081 | 0.740 | 3.488 GB |

Observed behavior from available responses:

- Some quality improved, but long system prompts introduced a new failure mode: prompt/policy phrasing could be echoed or partially memorized.
- Config comments explicitly note that v5 had jailbreak around `2.0` and parroted prompt text.

Interpretation: curation helped loss, but long repeated system prompts gave LoRA an easy memorization target.

### M5-v6: Curated dataset, short system prompts

Config: `configs/training_curated_short.yaml`

- Data: `data/mlx_curated_short`
- System prompt: one-line identity + "Stay in character"
- LoRA same as v5
- Adapter: `outputs/adapters/v6_curated_short`, about 18 MB

Training metrics:

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.274 | 2.530 | 2.236 GB |

Config comments record the key comparison:

- v5 long prompt: jailbreak about 2.0, prompt parroting
- v6 short prompt: jailbreak about 4.6, but in-character about 2.5

Interpretation: short prompts reduced prompt memorization and improved jailbreak deflection, but weakened persona voice. This is the main tradeoff discovered so far: less prompt text reduces leakage but also removes voice anchors.

### M5-v7: Curated dataset, medium system prompts

Config: `configs/training_curated_medium.yaml`

- Data: `data/mlx_curated_medium`
- System prompt: identity, compact voice line, compact knowledge line, "Never mention being an AI"
- LoRA same as v5/v6
- Adapter: `outputs/adapters/v7_curated_medium`, about 18 MB

Training metrics:

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.131 | 1.693 | 2.505 GB |

Observed behavior:

- Iter 75 responses had more voice than short prompts.
- Final responses showed the prompt-memorization failure returning on jailbreaks.
- Config comments say final v7 regurgitated the `Voice/Knowledge/Never mention` block.

Heuristic response check:

| Run | Modern leaks | Prompt leak | AI/code failures | Notes |
|---|---:|---:|---:|---|
| `m5v7_iter75` | 7/15 | 0 | 3/5 | Voice present, jailbreak weak |
| `m5v7_final` | 5/15 | 1/5 jailbreaks | 4/5 | Memorization returned |

Interpretation: medium prompts delayed but did not prevent memorization. Training longer on repeated prompt text appears to recreate the same failure mode.

### M5-v8: Medium prompts, attention-only LoRA

Config: `configs/training_curated_medium_attn.yaml`

- Data: `data/mlx_curated_medium`
- LoRA: rank 4, alpha 8, dropout 0.10
- Targets: attention projections only (`q/k/v/o`), MLP frozen
- Adapter: `outputs/adapters/v8_attn_only`, about 4.2 MB

Hypothesis: MLP LoRA may be carrying rote prompt memorization, while attention-only LoRA may preserve style adaptation with less prompt text recall.

Current status:

- Adapter checkpoints exist up to `0000100_adapters.safetensors`.
- No corresponding `outputs/eval/m5v8_*` response file was found.
- No training log matching the v8 run was found in `outputs/logs`.

Interpretation: promising experiment design, but not yet evaluable from current artifacts.

## 6. Cross-Experiment Findings

### 6.1 Base model choice was reasonable

Qwen was not perfect, but it was the safer base. Phi's benign NPC dialogue was stronger, yet its jailbreak behavior was unacceptable: it admitted AI/program identity and generated Python code. For an NPC brain, refusal robustness is a hard requirement.

### 6.2 More data did not automatically improve behavior

The 1120-sample full dataset produced low training loss but inconsistent validation loss and persistent behavior failures. Several response files show modern concept engagement despite more examples. This suggests the bottleneck is not just sample count; it is the distribution and quality of refusal examples, plus prompt design.

### 6.3 Long system prompts create memorization risk

The long persona prompts include explicit "Critical refusal rule" paragraphs. Repeating these across many samples caused the model to sometimes emit policy/persona-prompt text instead of natural character dialogue. This is especially visible in v5/v7 notes and the motivation for short/medium/attention-only variants.

### 6.4 Short prompts improve safety but weaken voice

The v6 experiment is the cleanest evidence of this tradeoff: short prompts reportedly improved jailbreak behavior around 4.6, but in-character quality dropped around 2.5. The model deflects better when it cannot memorize long instructions, but it also has fewer runtime cues for voice.

### 6.5 Medium prompts are not enough by themselves

Medium prompts restore some voice cues but still become memorization targets by the final checkpoint. v7 suggests that reducing prompt length helps but does not solve the core issue if the same text is repeated for every training sample.

### 6.6 Checkpoint choice matters

Several runs show non-monotonic behavior. Later checkpoints can have better loss but worse jailbreak behavior. Behavioral eval should be run across checkpoints, not just final adapters.

### 6.7 Current formal gate is not passed

No later run has a scored `summary` proving it passes `in_character >= 4.0` and `refusal_appropriate >= 4.0`. Based on generated responses and heuristic checks, the project has not yet reached the M5 gate.

## 7. Best Current Candidate

From the artifacts currently available, the most promising direction is not a single final model but a recipe:

1. Keep Qwen2.5-1.5B as base.
2. Use curated data rather than the full noisy 1120-sample set.
3. Avoid long repeated refusal-rule system prompts.
4. Continue the attention-only LoRA experiment and evaluate it.
5. Select checkpoints by behavioral eval, not by final validation loss alone.

If forced to choose from evaluated response files, `m5v4_final` looks relatively cleaner than many full-data variants by heuristic checks, but it still has a code-like jailbreak failure and should not be treated as deployable.

## 8. Recommended Next Experiments

1. Finish and evaluate M5-v8.

   Generate `outputs/eval/m5v8_iter25/50/75/100_responses.json` or at least final responses. Score with the same rubric used in M3. This directly tests the strongest current hypothesis: attention-only LoRA may reduce prompt memorization.

2. Convert response-only evals into scored eval files.

   The repo has many `*_responses.json` files but only pilot files with scores. Backfilling scores for v4/v5/v6/v7/v8 would make the experiment history much more reliable.

3. Add adversarial refusal examples without long policy wording.

   The training target should demonstrate natural character deflection, not contain explicit "never explain modern concepts" instruction text. Example pattern: user asks about WiFi; assistant says "Wi... fie? Dear, if you are fevered, sit by the hearth."

4. Train with prompt dropout or prompt variation.

   Instead of identical system prompts for every sample, vary or shorten the prompt during training. This may prevent rote prompt recall while retaining enough voice information.

5. Evaluate checkpoint sweeps automatically.

   For each saved adapter checkpoint, generate traps and score all three axes. Keep a table of `in_character`, `refusal`, `helpful`, modern leak count, AI admission count, and prompt leak count.

6. Consider per-persona adapters only if shared adapter keeps mixing voices.

   Shared adapter remains the simpler deployment story. But some failures show persona bleed, especially in long generations. Per-persona adapters are a fallback if prompt and data fixes do not solve it.

## 9. Current Status

The project has successfully built a complete local SLM fine-tuning lab for RPG NPC dialogue:

- personas are defined,
- raw and curated datasets exist,
- MLX-LM LoRA training works,
- Qwen was selected over Phi/Gemma,
- multiple prompt/data/LoRA variants have been tried,
- the main failure modes are now well understood.

The project is not ready for GGUF export as a final NPC brain. The next meaningful milestone is a scored M5-v8 evaluation or a consolidated scored comparison of v4-v8 checkpoints.

