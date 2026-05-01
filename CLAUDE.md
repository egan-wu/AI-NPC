# CLAUDE.md

Context for future Claude sessions in this repo. Read this first.

## What this project is

QLoRA fine-tuning of a Small Language Model to act as RPG NPC brains
(fantasy world, English dialogue). Final artifact is a 4-bit GGUF file
loadable by `llama.cpp`. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full design.

## Source of truth

`PROJECT_PLAN.md` is the design doc. If a request conflicts with it, surface
the conflict before coding — don't silently deviate. If the user agrees to a
change, update `PROJECT_PLAN.md` in the same turn.

## Environment

- **Phase 1 training: macOS + Apple Silicon, MLX-LM**. Do not suggest
  Unsloth/bitsandbytes commands until the user moves to Phase 2 (NVIDIA).
- Data generation: two valid paths —
  (1) **In-session authoring by Claude (preferred when iterating):** the
      assistant writes samples directly into `data/raw/*.jsonl` files in
      the current session, no API calls or API key needed. Use this for
      smaller targeted batches and quality iteration.
      **IMPORTANT:** when this path is used, the active Claude model
      MUST be Haiku-class (`claude-haiku-4-5-20251001` or successor).
      If the user is running a larger model (Opus/Sonnet), pause and
      ask them to `/model claude-haiku-4-5-20251001` before drafting
      sample text. Architectural discussion/planning can stay on the
      larger model — only the sample drafting itself must be on Haiku.
  (2) **Bulk batched generation via Haiku API:** Claude Haiku 4.5
      (`claude-haiku-4-5-20251001`) via the Anthropic Python SDK with
      prompt caching for the persona + few-shot block. Use this when
      generating large batches non-interactively.
- **Evaluation: two valid paths —**
  (1) **In-session scoring (preferred):** run `04_generate_responses.py`
      locally (no API key) to produce a `*_responses.json` file, then
      read the responses directly in-session and score each one.
      No `ANTHROPIC_API_KEY` needed. Use this by default.
  (2) **Automated judge via API:** `04_eval.py` calls Claude Haiku as judge
      for each trap. Requires `ANTHROPIC_API_KEY` in `.env`. Use this only
      for large batch evals or CI where interactive scoring is impractical.
- **Guard classifier LLM judge (10_guard_tiered.py): in-session only.**
  Do NOT call the Haiku API for gray-zone judgments. Instead:
  (1) Run the script with `--list-gray-zone` to identify which inputs need judgment.
  (2) Claude reads them in-session and provides decisions as a JSON dict
      (`--judgments '{"idx": "ALLOW/BLOCK", ...}'`).
  Never add API-call mode to guard judge scripts.
- Python deps go in `requirements.txt`. Secrets in `.env` (gitignored);
  template in `.env.example`.

## Conventions

- **Dataset format**: JSONL, one sample per line, `messages` array with
  `system` / `user` / `assistant` roles. Never collapse persona into the
  user turn.
- **Personas**: defined in `configs/personas.yaml`. The `system_prompt` field
  there is the literal string used at training and inference time — keep them
  identical.
- **Hyperparameters**: live in `configs/training.yaml`. Don't hardcode in
  scripts. Don't change `r`, `alpha`, or `target_modules` without asking —
  the values in PROJECT_PLAN §5.2 are deliberate starting points.
- **Scripts are numbered** (`01_…`, `02_…`) to signal pipeline order. New
  scripts should follow the same numbering.

## Things to leave alone

- `data/raw/` — generated dialogue samples (Haiku-generated *or* in-session
  Claude-authored). Treat as immutable once written; never edit prior raw
  files. Augmentations go into new files (e.g. `*_v2.jsonl`),
  transformations go into `data/processed/`.
- `outputs/` — training artifacts. Don't delete checkpoints without asking.
- The seed examples once written — they're the anchor for data quality.

## Working style

- The user wants to discuss before implementing on non-trivial changes.
  Default to proposing, then waiting for confirmation.
- Keep scripts small and single-purpose. No mega-scripts that do generation
  + training + eval in one file.
- Comments only where the *why* is non-obvious (e.g. why a specific LoRA
  target module list, why a particular tokenizer quirk matters).
