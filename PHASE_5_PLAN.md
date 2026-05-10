# Phase 5 — KV Cache Inference Optimization Plan

> **問題根源：** 每次使用者輸入，整個 prompt（system + mem_context + history + new_input）
> 都從零開始 Prefill，時間複雜度為 O(N)，N 為 prompt token 總數。
>
> **解決核心：** 從「無狀態計算」轉向「狀態機儲存」——
> 將重複的矩陣運算（Prefill）轉化為記憶體狀態（KV Cache）的重用，
> 將時間複雜度從 O(N) 降至 O(1)。

---

## 1. 問題診斷

### 1.1 實測基準數據

```
[guard=197ms | mem=17ms | ttft=4959ms | gen=9.2s | total=9.4s]
```

| Stage | 時間 | 說明 |
|-------|------|------|
| Guard (embedding + classify) | ~197 ms | SentenceTransformer encode 為主 |
| Memory retrieval (ChromaDB) | ~17 ms | 共享 embedding 後已優化 |
| **TTFT (Prefill)** | **~5000 ms** | **主要瓶頸** |
| Decode (token generation) | ~4 s | 正常，~40 tok/s |

### 1.2 TTFT 瓶頸分析

```
Prefill 速度（Apple Silicon, Mistral-7B 4-bit）：~208 tokens/s

目前每輪 prompt 組成：
  system_prompt   ≈ 250 tokens  ← 每輪相同，卻每次重算
  mem_context     ≈ 100 tokens  ← 每輪不同
  history         ≈ 50×N tokens ← 隨對話增長
  new_user_input  ≈  20 tokens
  ─────────────────────────────
  total           ≈ 730 tokens  → 730 / 208 ≈ 3.5s → 實測 ~5s TTFT
```

---

## 2. 現有 Prompt 拓撲結構（問題根源）

每次對話都完整執行以下 Prefill：

```
Turn N 的完整 prompt：

[INST] {system_prompt}                ← ~250 tokens，靜態，每輪重算 ❌
       + {mem_context}                ← ~100 tokens，動態
       + {turn_1_user}      [/INST]   ← 歷史，每輪重算 ❌
{turn_1_npc}
[INST] {turn_2_user}        [/INST]   ← 歷史，每輪重算 ❌
{turn_2_npc}
...
[INST] {new_user_input}     [/INST]   ← 本輪新增
```

**問題：** `mem_context` 注入在 system block 最前面，
每輪因為 query 不同而改變 → 整個 KV Cache 從頭失效。

---

## 3. 三種優化方案

### 方案 α — Static Prefix Cache（靜態前綴快取）

**核心思維：** 將永不改變的 `system_prompt` 在角色選定時預先 Prefill，
存成 KV Cache 物件，每輪對話直接繼承。

```
角色選定時（一次性）
    │
    ▼
prefill: [INST] {system_prompt} [/INST]
    │
    ▼
KV Cache 物件 ──→ 每輪重用

每輪 delta prefill（仍需重算）：
    {mem_context} + {history} + {new_user_input}
    ≈ 730 - 250 = 480 tokens → TTFT ≈ 2.3s
```

| 項目 | 數值 |
|------|------|
| 省掉的 tokens | ~250（system_prompt） |
| 預期 TTFT | ~3.5s → **~2.3s** |
| 實作難度 | ⭐⭐ |
| 核心技術 | `mlx_lm.utils.make_prompt_cache` |

---

### 方案 β — Incremental Cache（增量式快取）⭐ 建議主路徑

**核心思維：** 重組 Prompt 拓撲結構，確保「變動資料」永遠在序列最後方。
KV Cache 只需往後延伸，永不失效。

#### Prompt 拓撲重組

**重組前（目前）：**
```
[INST] system + mem_context + turn1_user [/INST] turn1_npc
[INST] turn2_user [/INST] turn2_npc
[INST] new_user [/INST]
         ↑
    mem_context 在最前，每輪改變 → 整個 cache 失效
```

**重組後（目標）：**
```
[INST] system_prompt [/INST] opening     ← 靜態，只算一次
[INST] turn1_user    [/INST] turn1_npc   ← 歷史，累積保留於 cache
[INST] turn2_user    [/INST] turn2_npc   ← 歷史，累積保留於 cache
[INST] [Mem: mem_context]                ← 動態，永遠在尾端 ✅
       new_user_input [/INST]
```

#### 每輪執行流程

```
角色選定時
    │
    ▼
prefill: [INST] {system_prompt} [/INST] {opening}
    │
    ▼
cache_0（靜態前綴，永久保留）
    │
    ├── Turn 1 ──────────────────────────────────────────────
    │   delta prefill: [INST] [Mem: mem1] + user1 [/INST]
    │   ≈ 150 tokens → TTFT ≈ 0.7s                          ✅
    │   generate → npc1
    │   cache_1 = cache_0 + [turn1 KV]
    │
    ├── Turn 2 ──────────────────────────────────────────────
    │   delta prefill: [INST] [Mem: mem2] + user2 [/INST]
    │   ≈ 150 tokens → TTFT ≈ 0.7s  （不隨對話增長）        ✅
    │   generate → npc2
    │   cache_2 = cache_1 + [turn2 KV]
    │
    └── Turn N ──────────────────────────────────────────────
        delta prefill: [INST] [Mem: memN] + userN [/INST]
        ≈ 150 tokens → TTFT ≈ 0.7s  （恆定）                ✅
```

| 項目 | 數值 |
|------|------|
| 每輪 delta tokens | ~150（mem_context + new_input） |
| 預期 TTFT | ~5.0s → **~0.7s**（提升 ~7×） |
| 實作難度 | ⭐⭐⭐ |
| 核心技術 | `make_prompt_cache` + `build_prompt()` 拓撲重組 |
| 主要代價 | 需繞過 `apply_chat_template` 全量重建，手動管理 `[INST]` 邊界 |

---

### 方案 γ — Pre-baked KV Cache（預烘焙快取）

**核心思維：** 針對「深厚故事背景」的冷啟動延遲。
以 I/O 吞吐取代 NPU 運算。

```
離線階段（部署前）
    │
    ▼
prefill 大量世界觀設定（數千 tokens 的 lore, history, faction data...）
    │
    ▼
KV Cache → 序列化 → .safetensors 檔案（存於 SSD）

連線階段（玩家讀取 NPC 時）
    │
    ▼
從 SSD 載入 .safetensors → 還原 KV Cache 至記憶體
    │
    ▼
跳過所有 Prefill 計算，直接從 cache offset 繼續
```

| 項目 | 說明 |
|------|------|
| 適用場景 | world_knowledge 擴充至 50+ 條、數千 token 的深度背景 |
| 預期效益 | 萬級 Token 背景：數秒運算 → **毫秒級 I/O** |
| 實作難度 | ⭐⭐⭐⭐ |
| 技術挑戰 | MLX KV Cache 序列化 API、版本失效管理（模型更新時 cache 需重建）|
| 現階段必要性 | ❌ 目前 world_knowledge 僅 8 條，不構成瓶頸，**列為未來選項** |

---

## 4. 方案比較總表

| 優化層級 | 方案 | 解決痛點 | TTFT 目標 | 實作難度 | 核心技術 |
|----------|------|----------|-----------|----------|----------|
| **現狀** | — | — | ~5.0s | — | 每輪全量 Prefill |
| **Opportunity 1** | α Static Prefix | 固定角色設定重複計算 | ~2.3s | ⭐⭐ | Static Prefix Cache |
| **Opportunity 2** | β Incremental | 對話越長 TTFT 越大 | **~0.7s** | ⭐⭐⭐ | Prompt 結構重組 |
| **Pre-baking** | γ Pre-baked | 深厚背景冷啟動 | ~ms（I/O） | ⭐⭐⭐⭐ | KV Cache 序列化 |

```
TTFT 視覺化比較：

現狀       ████████████████████████  5.0s  每輪全重算 ~730 tokens
方案 α     █████████████████         2.3s  省 system_prompt ~250 tokens
方案 β     ██                        0.7s  每輪只算 ~150 tokens delta
方案 γ     （解決不同問題：大背景冷啟動，目前規模不適用）
```

---

## 5. 建議執行路徑

```
Phase 5.1  →  實作方案 β（含 α 效果）
               ├─ 修改 build_prompt() 拓撲結構
               │    mem_context 從 system block 移至每輪 [INST] 尾端
               ├─ 角色選定時 make_prompt_cache(model)
               │    prefill: [INST] system_prompt [/INST] opening
               │    存入 cache 物件
               └─ 每輪 stream_generate(delta, prompt_cache=cache)
                    delta = [Mem: mem_context] + new_user_input

Phase 5.2  →  驗證與調校
               ├─ TTFT 基準測試（對比 5.0s 基準）
               ├─ 確認 mem_context 位置改變後 NPC 回應品質不退化
               └─ --timing 輸出更新（加入 cache hit 資訊）

Phase 5.3  →  視需求評估方案 γ
               條件：world_knowledge 擴充至 50+ 條，或引入完整 lore 文件
```

---

## 6. 實作變動點

### 方案 β（已完成）

| 檔案 | 變動 |
|------|------|
| `scripts/20_npc_cli_memory.py` | 移除 `build_prompt()`，新增 `build_turn_delta_tokens()`；`chat_loop` 建立 cache 並在 opening 時填入；每輪傳 delta tokens + `prompt_cache=cache`；`--no-history` 以 `trim_prompt_cache` 實作 |
| `src/memory_module.py` | 無需變動（`retrieve_context` API 不變） |

關鍵新函式：
```python
def build_turn_delta_tokens(tokenizer, mem_context, user_input, jailbreak_note="") -> list[int]:
    # [Critical instruction] (if jailbreak)
    # [Memory context for this response] (if mem_context)
    # user_input   ← 緊鄰 [/INST]，recency 最高
    text = f"[INST] {content} [/INST]"
    return tokenizer.encode(text, add_special_tokens=False)  # 無 BOS
```

timing tag 新增 `cache=Ntok` 欄位顯示累積 cache 長度。

---

### 方案 γ（已完成）

| 檔案 | 說明 |
|------|------|
| `src/cache_utils.py` | 新增。`save_cache` / `load_cache` / `prebaked_path` / `is_valid` |
| `scripts/22_prebake_cache.py` | 新增。離線烘焙腳本，`-p <persona>` 或 `--all` |
| `scripts/20_npc_cli_memory.py` | 新增 `--prebaked-cache auto/on/off`；`chat_loop` 接受 `prebaked_cache_file` 參數；γ path 從 SSD 載入 cache + 顯示存檔 opening |

Pre-baked 內容：
```
[INST] system_prompt

[World Knowledge — all facts]
- fact_1 … fact_N

opening_cue [/INST] opening_response </s>
```

使用流程：
```bash
# 離線烘焙（一次性）
python scripts/22_prebake_cache.py -p marta

# 自動使用（預設 auto）
python scripts/20_npc_cli_memory.py -p marta

# 強制使用 / 強制略過
python scripts/20_npc_cli_memory.py -p marta --prebaked-cache on
python scripts/20_npc_cli_memory.py -p marta --prebaked-cache off
```

---

## 7. 技術風險與注意事項

1. **Mistral 無 system role**：system_prompt 原本塞在第一個 `[INST]` 內，
   分割時需手動控制 token 邊界，繞過 `apply_chat_template` 全量重建。

2. **mem_context 位置效果驗證**：將 mem_context 從 system block 移至每輪 `[INST]`
   後，需測試模型是否同樣有效參考該資訊（理論上 attention 可達，實測為準）。

3. **Cache 物件生命週期**：`--fresh` 重設對話時，cache 應回退到靜態前綴狀態
   （system_prompt + opening），而非完全清除。

4. **max_kv_size 上限**：長對話可能超出 KV Cache 最大容量，需設定
   `max_kv_size` 並處理 cache trim 或 sliding window。
