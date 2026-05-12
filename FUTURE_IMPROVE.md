# FUTURE_IMPROVE.md — Roadmap & Known Gaps

Living document. Each item has:
- **Why** — the user-visible symptom or design risk
- **Direction** — concrete fix path (not yet a plan)
- **Impact / Effort** — 1–5 each (5 = highest); ⭐⭐⭐ marks the high-ROI items

Items are grouped by track. When an item gets picked up, move it to a phase
plan (`PHASE_X_PLAN.md`) and link the resulting commit here.

---

## 0. Critical findings (verified during 2026-05-10 review)

### CR1 — LoRA adapters never reached interactive CLI ✅ (resolved 2026-05-12)

`scripts/19_npc_cli.py`, `20_npc_cli_memory.py`, `22_prebake_cache.py` all
called `mlx_lm.load(MODEL_ID)` with **no `adapter_path`**. Every "Marta"
session before commit `dba6925` was **base Mistral-7B-Instruct-4bit
steered only by `system_prompt`** — none of the trained adapters under
`outputs/adapters/` had ever been live.

**Status**
- **2026-05-10** — F1 wiring landed in commit `dba6925`. `--adapter PATH`
  flag added to both 20_npc_cli_memory.py and 22_prebake_cache.py; γ cache
  is now keyed on (npc, model, adapter) and `is_valid()` rejects adapter
  mismatches.
- **Smoke-test result**: loading `mistral_iter100` works mechanically but
  produces degraded output ("Roughly a foot and a hand, both shaped like
  an hourglass…"). Root cause: this adapter was trained on
  `data/mlx_mistral` (raw v1 samples, non-curated, voice-mismatched).
- **Next**: retrain on the new 300-sample curated set
  (`data/processed/train_curated.jsonl`). Config and prep script staged:
  - `configs/training_curated_mistral.yaml`
  - `data/mlx_curated_mistral/{train.jsonl 270, valid.jsonl 30}` (stratified per persona)
  - `scripts/prep_curated_mistral.py`
  Run with: `python scripts/03_train_mlx.py --config configs/training_curated_mistral.yaml`
  Expected: 30–45 min on Apple Silicon, 6 checkpoints at iter 25/50/…/150.

- **2026-05-12** — Training run #1 (lr=2e-4) exploded — val loss 3.27 → 10.96
  at iter 25. Root cause: 2e-4 extrapolated from Qwen 1.5B was too high for
  Mistral 7B. Killed at iter 50.
- **2026-05-12** — Training run #2 (lr=5e-5) **succeeded**. Val loss:
  iter 50/75/100/125/150 → 0.405/0.404/0.400/**0.397**/0.410. Best
  checkpoint **iter 125** (val 0.397). Iter 150 starts overfitting
  (val rebounds while train continues descending to 0.254).
  Canonical adapter saved at `outputs/adapters/curated_mistral_iter125/`.
- **2026-05-12** — Interactive smoke test confirms LoRA voice is excellent
  with all mem layers OFF (`--k-world 0 --k-player 0 --k-persona 0 --k-conv 0`):
  - "Aye, *nods slowly* I'm Marta—the innkeeper..."
  - "Wi-what? You're speaking oddly. I think you've had one too many at the
    merchant's stall, mate..." (anachronism handling perfect)
  However, with γ cache + L0/L3 baked in, Marta confuses her identity with
  Wenric/Garrick. This is the **B3 memory-bleed** issue (LoRA trained on
  `system + user → assistant` only — never saw labelled `[World]/[About me]`
  context blocks). B1 is functionally done; B3 is the next gate.

---

## 1. System Functionality

| # | Gap | Direction | Impact / Effort |
|---|-----|-----------|-----------------|
| **F1** | LoRA adapter not wired to CLI (= CR1) | `20_npc_cli_memory.py` add `--adapter PATH`; pass to `mlx_lm.load(MODEL_ID, adapter_path=...)`; **γ cache must be invalidated when adapter changes** — add `adapter_id` to cache metadata + `is_valid()` check | **5 / 2** ⭐⭐⭐ |
| **F2** | No multi-NPC switching mid-session | CLI command `/switch <npc>`; hold a `dict[npc_id, kv_cache]` so switching is just swapping the cache pointer (no re-prefill) | 4 / 3 |
| **F3** | No save-game / multi-save support | `--save <name>` flag; isolate per save: `outputs/saves/{name}/chroma_db` + `player_lore.json` snapshot | 3 / 3 |
| **F4** | GM mode missing — adding player_lore needs another terminal AND has cross-process refresh issue (P6) | CLI built-in commands `/lore add <text>` / `/lore list` / `/lore rm <n>`; same-process write avoids the Chroma HNSW staleness entirely | **5 / 1** ⭐⭐⭐ |
| **F5** | `world_knowledge.yaml` only 14 facts — world feels thin | Extract common lore from existing 5 personas' `system_prompt`, expand to ~50 facts (human-curate); re-seed L0 + re-bake γ | 4 / 2 |
| **F6** | No conversation export / replay | `--save-transcript <path>` writes JSONL; small `25_replay.py` to render sessions | 2 / 2 |
| F7 | ChromaDB cross-process write not visible to running session | Subsumed by F4 — same-process writes side-step the issue | 3 / 0 (with F4) |

---

## 2. System Optimization

| # | Gap | Direction | Impact / Effort |
|---|-----|-----------|-----------------|
| **P1** | KV cache has no upper bound — long sessions risk OOM / slow attention | Set `max_kv_size=4096`; sliding-window trim oldest dialogue turns while preserving the static prefix (cache[0..static_cache_len]) | 4 / 2 |
| P2 | Each turn issues 4 separate ChromaDB queries (~17ms × 4) | Profile first; if real bottleneck, batch-query or merge into one collection with metadata filter | 2 / 3 |
| **P3** | Model load takes ~2s on every session restart | Run as a daemon (FastAPI / unix socket); CLI becomes a thin client; multiple sessions share weights | 4 / 4 |
| P4 | No retrieval re-ranking — top-k by pure cosine sim drags in look-alike-but-irrelevant facts (PROJECT_PLAN A1) | Fetch top-2k, rerank with MMR (Maximal Marginal Relevance) or a small cross-encoder; trim to k | 3 / 3 |
| P5 | `player_lore` has no recency boost — old facts can outrank fresh ones | metadata already has `timestamp`; blend: `score = 0.7 * sim + 0.3 * recency` | 2 / 2 |
| P6 | Cross-process L_p refresh (acknowledged) | Deferred — solved indirectly by F4 (in-process /lore add) | — |

---

## 3. AI NPC Behaviour Quality

| # | Gap | Direction | Impact / Effort |
|---|-----|-----------|-----------------|
| ~~B1~~ | ✅ DONE 2026-05-12 — curated_mistral_iter125 adapter delivers strong voice ("Aye, *nods slowly* I'm Marta") in pure-LoRA mode | Next: B3 (memory bleed under labelled context) | — |
| **B2** | NPC quotes retrieved facts verbatim — "Stag and Thistle has eight rooms, a common room and a stable" — breaks immersion | (a) `system_prompt` add: "Use facts as inspiration; never quote them verbatim"; (b) train 10–20 paraphrase-style samples; (c) optionally LLM-rewrite retrieved facts before injection (latency cost) | 4 / 3 |
| **B3** | Memory bleed — LoRA trained on `system + user → assistant` only. With γ cache (L0+L3 baked) Marta confuses her own identity with Wenric/Garrick because she's never seen labelled `[World]/[About me]` blocks. Verified 2026-05-12. Three paths: **(A) `system_prompt` guardrail** ("Facts below are YOUR memories; you remain Marta") — cheapest, try first. (B) Retrain LoRA with 5–10 mem-aware samples — most principled. (C) Gating: inject L0/L3 only when query has keyword overlap. | **5 / 2** ⭐⭐⭐ NEXT |
| B4 | No emotional / relationship state — Marta is identical to a regular vs a stranger | `relationship_state.yaml` per-NPC: `{trust: 0–100, mood: ...}`; LLM updates after each turn; injected into per-turn delta; player actions adjust trust | 5 / 4 |
| B5 | Anachronism reactions identical across personas (Lily and Wenric should react very differently) | Training data already encodes this (v3/v9 samples differ); base model loses it — solving B1 should fix this for free | 3 / 0 (with B1) |
| **B6** | No graceful "I don't know" — retrieval miss leads to confabulation | (a) `system_prompt`: "If the answer isn't in the facts above, admit you don't know in character"; (b) seed 5–10 grounding samples in training data | 4 / 2 |
| **B7** | Eval set is 30 traps total — too narrow to measure B1–B6 improvements | Expand to 100+ traps, layered by category: voice / lore-faithfulness / memory-recall / anti-paraphrase; LLM-as-judge automation | 4 / 3 |
| B8 | No multi-turn coherence test — eval is single-turn only | When expanding eval, add 5-turn dialogues; judge whether turn N stays consistent with turn 1 | 3 / 3 |

---

## Suggested Sprint Order

### Sprint 1 — High-ROI unlock
1. **F1 + B1 (wire LoRA)** — single change unblocks Phase 2's value
2. **F4 (in-CLI /lore command)** — fixes F7 + P6 as a side-effect
3. **F5 (expand L0 to ~50 facts)** — half-hour effort, world thickens

### Sprint 2 — Behaviour fixes (assumes Sprint 1 done so changes are measurable)
4. **B6 (graceful "I don't know")** — easy `system_prompt` change + a few samples
5. **B3 (memory bleed)** — likely needs gating + prompt tweak
6. **B2 (verbatim-quote suppression)** — depends on whether LoRA already helps

### Sprint 3 — Measurement infrastructure
7. **B7 + B8 (expand eval, multi-turn coherence)** — without this, all later optimisation is gut-feel

### Sprint 4 — Functional expansion
8. **F2 (NPC switching)** — depends on demand
9. **F3 (save-game)** — when multi-user testing arrives

### Sprint 5 — Performance & deep features
10. **P1 (max_kv_size cap)** — needed once long sessions are common
11. **P3 (daemon)** — when demoing or supporting multiple concurrent sessions
12. **B4 (emotion / relationship state)** — biggest design change; do last

---

## Out of scope (deferred indefinitely)

- **L1 faction / L2 region memory layers** — gated on personas.yaml acquiring
  `factions:` and `region:` fields; only valuable if NPC count grows beyond 5.
- **Cross-NPC episodic propagation** — "Marta heard from Roderick"; needs
  gossip-spread rules and is much harder than L_p (which is "all NPCs know").
- **Auto-detect player deeds** — post-conversation hook proposing player_lore
  entries; manual entry remains the safer default.

---

## Change log

- **2026-05-10** — initial draft after Phase 6 landing; CR1 LoRA gap
  identified during this review.
- **2026-05-11** — F1 wiring landed (`dba6925`); CR1 status updated.
  mistral_iter100 confirmed broken (raw v1 data contamination).
  Curated Mistral retrain staged (config + prep script + data split ready).
  B1 awaiting training execution.
- **2026-05-12** — B1 done. curated_mistral_iter125 trained on the
  300-sample curated set delivers excellent voice in pure-LoRA mode.
  New gate identified: B3 memory bleed — labelled mem-hierarchy context
  confuses identity. Promoted to next ⭐⭐⭐ task. Path A (system_prompt
  guardrail) recommended first.
