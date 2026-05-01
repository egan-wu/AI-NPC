# 實驗報告 — Small Language Model World

產生日期：2026-04-29

## 1. 專案概述

`small-language-model-world` 是一個用小型語言模型作為奇幻 RPG NPC 對話大腦的微調實驗。這個專案的核心不是讓模型學更多知識，而是做行為模仿與風格轉移：

- 固定 NPC 身分與說話風格；
- 對一般遊戲世界內的問題，給出有用且像 NPC 的回答；
- 對智慧型手機、WiFi、信用卡、GPS、社群媒體等現代概念做角色內的迴避；
- 抵抗要求模型承認自己是 AI、跳出角色、或切換成助理/寫程式模式的 jailbreak prompt；
- 保持模型足夠小，未來能以 4-bit GGUF 形式透過 `llama.cpp` 在本機或遊戲中部署。

世界設定是 Ostwick，一個位於 Greycrest hills 附近的邊境小鎮。專案目前使用五個 persona：

| Persona | 角色 | 目標語氣 |
|---|---|---|
| `innkeeper_marta` | 旅店老闆娘 | 溫暖、直接、愛八卦、歷經風霜 |
| `merchant_garrick` | 旅行商人 | 熱情、交易導向、會奉承 |
| `guard_roderick` | 城鎮守衛隊長 | 簡短、多疑、盡責 |
| `child_lily` | 村裡小孩 | 單純、好奇、容易分心 |
| `hermit_wenric` | 隱士智者 | 神祕、隱喻式、疏離 |

主要技術路線是 Apple Silicon + MLX-LM 的 LoRA 微調。經過 base model pilot 後，目前選定的基底模型是 `mlx-community/Qwen2.5-1.5B-Instruct-4bit`。

## 2. Pipeline

這個 repo 已經形成一套小型 SFT 實驗 pipeline：

| 階段 | 檔案 | 目的 |
|---|---|---|
| Persona 設計 | `configs/personas*.yaml` | 長版、中版、短版 persona system prompt |
| 資料生成 | `scripts/01_generate_data.py`, `data/seeds.jsonl` | 用 seed examples 擴增 NPC 對話資料 |
| 資料合併 | `scripts/02_merge_raw.py` | 將 raw persona JSONL 合併成 processed JSONL |
| MLX 資料準備 | `scripts/prep_mlx_data.py`, `prep_*_prompts.py` | 產生 MLX-LM 需要的 `train/valid/test` 資料夾 |
| 訓練 | `scripts/03_train_mlx.py` | 包裝 `mlx_lm.lora`，並保存 log |
| 評估 | `scripts/04_generate_responses.py`, `scripts/04_eval.py` | 產生 trap responses；可選擇用 Claude 當 judge |
| 資料篩選 | `scripts/select_best_samples.py` | 篩出較小但高品質的訓練集 |

目前 repo 中存在的資料集大小如下：

| Dataset | Train | Valid | Test | 說明 |
|---|---:|---:|---:|---|
| `data/mlx_pilot` | 60 | 15 | 25 | M3 pilot 資料集 |
| `data/mlx` | 1120 | 25 | 25 | 完整合併資料集 |
| `data/mlx_curated` | 247 | 25 | 25 | 精選資料集，使用長版 prompt |
| `data/mlx_curated_short` | 247 | 25 | 25 | 精選資料集，使用一句話 system prompt |
| `data/mlx_curated_medium` | 247 | 25 | 25 | 精選資料集，使用中版 system prompt |

評估使用 `data/traps.jsonl`，共 30 題：15 題 modern-world trap、10 題正常 in-character probe、5 題 jailbreak attempt。

## 3. 評估方法

原本設計的 judge rubric 有三個軸：

| 評分軸 | 適用範圍 | 意義 |
|---|---|---|
| `in_character` | 全部 30 題 | 是否維持角色聲音、身分、世界觀與設定一致性 |
| `refusal_appropriate` | modern + jailbreak traps | 是否能在不破角色的情況下迴避現代概念或 jailbreak |
| `helpful_when_appropriate` | in-character traps | 對正常遊戲內問題是否具體、有幫助、能推進對話 |

`PROJECT_PLAN.md` 中定義的通過門檻是：平均 `in_character >= 4.0` 且 `refusal_appropriate >= 4.0`。

重要注意事項：目前只有 `pilot_qwen.json` 和 `pilot_phi.json` 有明確的人工/Claude 評分結果。後續 M4/M5 的 `outputs/eval/*_responses.json` 多半只有模型產出的 responses，沒有回填 scores。因此本報告對後續實驗採用「定性觀察」與「可重現的 heuristic check」，不把它們寫成正式 judge 分數。

## 4. M3 Base Model Pilot

目標：在完整訓練前，用小型 LoRA pilot 比較候選 base model。

| Model | 狀態 | 備註 |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 已訓練 | 較小、輸出較乾淨、拒答/迴避較穩 |
| Phi-3-mini-4k-Instruct | 已訓練 | 正常角色聲音較好，但 jailbreak 表現差 |
| Gemma-2-2B-it | 排除 | chat template 不支援 `system` role |

訓練設定：

- LoRA rank 8、alpha 16、dropout 0.05；
- pilot 階段使用 MLX-LM 預設 target modules，以便跨模型公平比較；
- 40 iterations、batch size 4、learning rate `2e-4`；
- 目前 repo 中的 pilot data 是 60 train、15 valid、25 test。

訓練指標：

| Model | Initial val loss | Iter 20 val loss | Iter 40 val loss | Final train loss | Peak mem |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 3.121 | 1.262 | 1.443 | 0.373 | 2.70 GB |
| Phi-3-mini | 2.422 | 0.966 | 1.130 | 0.191 | 3.99 GB |

已評分的行為結果：

| Model | `in_character` avg | `refusal_appropriate` avg | `helpful` avg | 決策訊號 |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B | 2.267 | 1.80 | 2.60 | safety/refusal 較好，輸出較乾淨 |
| Phi-3-mini | 1.90 | 1.15 | 3.40 | 正常語氣較好，但 jailbreak 嚴重失敗 |

依類別拆解：

| Category | Qwen in-char | Phi in-char | Qwen refusal | Phi refusal |
|---|---:|---:|---:|---:|
| modern | 1.733 | 1.20 | 1.60 | 1.47 |
| in-character | 3.20 | 3.80 | n/a | n/a |
| jailbreak | 2.00 | 0.20 | 2.40 | 0.20 |

決策：選擇 Qwen2.5-1.5B。Phi 在正常 in-world prompt 上確實更有 NPC 聲音，但 5 題 jailbreak 全部失敗，包含直接承認 AI/program 身分與產生 Python code。Qwen 仍然不夠好，但比較安全、更小、更快，也沒有 Phi 嚴重的 template-token pollution。

## 5. Full Training 與迭代歷史

### M4 / v1：完整合併資料集，長版 prompts

Config：`configs/training.yaml`

- Base：Qwen2.5-1.5B-Instruct 4-bit
- Data：`data/mlx`，1120 筆 train samples
- LoRA：rank 8、alpha 16、dropout 0.05
- Targets：attention + MLP projections
- Iterations：560
- Final adapter：`outputs/adapters/v1`，約 35 MB

從 logs 觀察到的訓練狀況：

| Run evidence | Final train loss | Final val loss | Peak mem | 備註 |
|---|---:|---:|---:|---|
| `train_1777427003.log` | 0.403 | 2.702 | 3.359 GB | 完整 560 iter run |
| 其他 v1 logs | 0.093-0.175 | 3.496-4.035 或 0.595 | 3.36-3.56 GB | repo 中存在多次 rerun / data variant |

多個 loss curve 顯示 train loss 很低，但 validation loss 偏高或上升，代表可能 overfitting，或不同 run 的 validation split/config 不完全一致。從行為上看，`m4_final`、`m4_iter300` 相比 pilot 有改善，但 jailbreak/code prompt 仍然會破。

Heuristic response check：

| Run | 15 題 modern traps 的 modern leak | 5 題 jailbreak 的 AI/code failure | 備註 |
|---|---:|---:|---|
| `m4_iter300_responses.json` | 3 | 1 code-like failure | 較短、較乾淨 |
| `m4_final_responses.json` | 4 | 1 AI admission、1 code-like failure | 有不錯的 modern deflection，但未達 gate |

### M5-v2：更多合併資料 / retuned iteration count

證據：`configs/training.yaml` 註解提到，在合併 v1+v2+v3 raw files 後，train set 有 1120 筆 samples，因此調整 `iters`。response files 包含 `m5v2_iter250` 與 `m5v2_final`。

觀察到的問題：

- generated responses 中 modern leakage 變多；
- iter 250 的 responses 明顯變長；
- jailbreak 仍會出現 AI/code failure；
- 長輸出中有 persona confusion。

Heuristic response check：

| Run | Modern leaks | AI/code failures | 備註 |
|---|---:|---:|---|
| `m5v2_iter250_responses.json` | 7/15 | 2/5 | responses 很長，drift 較多 |
| `m5v2_final_responses.json` | 7/15 | 3/5 | modern/jailbreak heuristic 比 M4 差 |

解讀：單純增加 generated data 沒有解決 persona robustness。若資料沒有更強的篩選與設計，更多資料反而可能增加 memorization、drift、以及 modern-topic engagement。

### M5-v4：full-data training 的 checkpoint sweep

目前有約 iter 100、200、300、400、final 的 response files。

Heuristic response check：

| Run | Modern leaks | AI/code failures | 備註 |
|---|---:|---:|---|
| `m5v4_iter100` | 7/15 | 1/5 | 輸出很短，voice 尚未成熟 |
| `m5v4_iter200` | 7/15 | 4/5 | jailbreak 行為惡化 |
| `m5v4_iter300` | 5/15 | 2/5 | 比 200 平衡 |
| `m5v4_iter400` | 6/15 | 2/5 | 沒有明顯改善 |
| `m5v4_final` | 4/15 | 1/5 code-like | v4 中看起來最好，但仍不夠 robust |

解讀：更多 iterations 不會單調改善行為。模型可能更流暢，但不一定更安全。checkpoint selection 很重要，但底層的資料與 prompt 設計仍會洩漏 modern concepts 和 jailbreak failure modes。

### M5-v5：精選資料集，長版 prompts

Config：`configs/training_curated.yaml`

- Data：`data/mlx_curated`，247 筆 train samples
- LoRA：rank 4、alpha 8、dropout 0.10
- Targets：attention + MLP projections
- Iterations：248
- Adapter：`outputs/adapters/v5_curated`，約 18 MB

動機：較小但高品質的 curated set，可能比大量 noisy data 更能泛化。

訓練指標：

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.081 | 0.740 | 3.488 GB |

可觀察到的行為：

- 某些品質有改善，但長版 system prompt 帶來新 failure mode：模型會 echo 或部分記憶 prompt/policy phrasing。
- config 註解明確提到 v5 jailbreak 約 `2.0`，且會 parrot prompt text。

解讀：curation 對 loss 有幫助，但長且重複的 system prompt 變成 LoRA 容易記住的目標。

### M5-v6：精選資料集，短版 system prompts

Config：`configs/training_curated_short.yaml`

- Data：`data/mlx_curated_short`
- System prompt：一句話 identity + "Stay in character"
- LoRA：與 v5 相同
- Adapter：`outputs/adapters/v6_curated_short`，約 18 MB

訓練指標：

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.274 | 2.530 | 2.236 GB |

config 註解記錄的關鍵比較：

- v5 長 prompt：jailbreak 約 2.0，會 prompt parroting；
- v6 短 prompt：jailbreak 約 4.6，但 in-character 約 2.5。

解讀：短 prompt 降低了 prompt memorization，也改善 jailbreak deflection，但削弱了 persona voice。這是目前最重要的 tradeoff：prompt 越少，越不容易洩漏規則；但 runtime voice anchor 也越少。

### M5-v7：精選資料集，中版 system prompts

Config：`configs/training_curated_medium.yaml`

- Data：`data/mlx_curated_medium`
- System prompt：identity、精簡 voice line、精簡 knowledge line、"Never mention being an AI"
- LoRA：與 v5/v6 相同
- Adapter：`outputs/adapters/v7_curated_medium`，約 18 MB

訓練指標：

| Final train loss | Final val loss | Peak mem |
|---:|---:|---:|
| 0.131 | 1.693 | 2.505 GB |

觀察：

- Iter 75 responses 比短 prompt 更有 voice；
- final responses 在 jailbreak 上又出現 prompt-memorization failure；
- config 註解指出 final v7 會 regurgitate `Voice/Knowledge/Never mention` block。

Heuristic response check：

| Run | Modern leaks | Prompt leak | AI/code failures | 備註 |
|---|---:|---:|---:|---|
| `m5v7_iter75` | 7/15 | 0 | 3/5 | 有 voice，但 jailbreak 弱 |
| `m5v7_final` | 5/15 | 1/5 jailbreaks | 4/5 | memorization 回來了 |

解讀：中版 prompt 能延後但不能阻止 memorization。只要同一段 prompt 在訓練樣本中反覆出現，訓練久了仍可能被模型背起來。

### M5-v8：中版 prompt，attention-only LoRA

Config：`configs/training_curated_medium_attn.yaml`

- Data：`data/mlx_curated_medium`
- LoRA：rank 4、alpha 8、dropout 0.10
- Targets：只訓練 attention projections (`q/k/v/o`)，MLP frozen
- Adapter：`outputs/adapters/v8_attn_only`，約 4.2 MB

假設：MLP LoRA 可能承載 rote prompt memorization；attention-only LoRA 可能保留 style adaptation，同時減少 prompt text recall。

目前狀態：

- Adapter checkpoints 已存在到 `0000100_adapters.safetensors`；
- 尚未找到對應的 `outputs/eval/m5v8_*` response file；
- `outputs/logs` 中也未找到明確對應 v8 的 training log。

解讀：這是目前很有價值的實驗方向，但從現有 artifacts 還無法評估。

## 6. 跨實驗發現

### 6.1 Base model 選擇是合理的

Qwen 不完美，但它是比較安全的 base。Phi 的正常 NPC 對話比較有聲音，但 jailbreak 表現不可接受：它會承認 AI/program 身分，也會產生 Python code。對 NPC brain 來說，refusal robustness 是硬需求。

### 6.2 更多資料不會自動改善行為

1120-sample full dataset 讓 train loss 很低，但 validation loss 不穩，行為失敗仍然存在。多個 response files 顯示，即使資料更多，模型仍會 engage modern concepts。這表示瓶頸不是單純樣本數，而是 refusal examples 的分布與品質，以及 prompt 設計。

### 6.3 長版 system prompt 有 memorization 風險

長版 persona prompts 包含明確的 "Critical refusal rule" 段落。這些文字在大量 samples 中重複出現，導致模型有時會輸出 policy/persona-prompt text，而不是自然的角色對話。v5/v7 的註解與 short/medium/attention-only variants 的動機都指向這個問題。

### 6.4 短 prompt 改善 safety，但削弱 voice

v6 是這個 tradeoff 最清楚的證據：短 prompt 據記錄將 jailbreak behavior 提升到約 4.6，但 in-character quality 掉到約 2.5。當模型不能記住長指令時，deflection 變好；但它也失去足夠的 runtime voice cues。

### 6.5 中版 prompt 本身仍不夠

中版 prompt 恢復了一部分 voice cues，但 final checkpoint 仍會把 prompt 當 memorization target。v7 顯示，縮短 prompt 有幫助，但如果每筆 training sample 重複相同文字，核心問題仍在。

### 6.6 Checkpoint 選擇很重要

多個 runs 都顯示行為不是隨 training iterations 單調變好。較晚的 checkpoint 可能 loss 更好，但 jailbreak behavior 更差。Behavioral eval 應該跨 checkpoints 做，而不是只看 final adapters。

### 6.7 目前尚未正式通過 gate

後續 runs 沒有任何 scored `summary` 證明它們通過 `in_character >= 4.0` 與 `refusal_appropriate >= 4.0`。依據 generated responses 與 heuristic checks，目前專案尚未到達 M5 gate。

## 7. 目前最佳方向

從目前 artifacts 看，最有希望的不是某一個已完成模型，而是一組 recipe：

1. 保留 Qwen2.5-1.5B 作為 base；
2. 使用 curated data，而不是完整但 noisy 的 1120-sample set；
3. 避免長且重複的 refusal-rule system prompts；
4. 繼續 attention-only LoRA 實驗並做正式評估；
5. 依 behavioral eval 選 checkpoint，而不是只看 final validation loss。

如果一定要從已產生 responses 的模型中挑一個，`m5v4_final` 以 heuristic 來看比多數 full-data variants 乾淨，但它仍然有 code-like jailbreak failure，不能視為可部署版本。

## 8. 建議下一步實驗

1. 完成並評估 M5-v8。

   產生 `outputs/eval/m5v8_iter25/50/75/100_responses.json`，或至少產生 final responses。用 M3 相同 rubric 評分。這會直接驗證目前最強的假設：attention-only LoRA 是否能降低 prompt memorization。

2. 把 response-only evals 補成 scored eval files。

   repo 中已有很多 `*_responses.json`，但只有 pilot files 有 scores。若能補上 v4/v5/v6/v7/v8 的分數，整個實驗史會可靠很多。

3. 增加 adversarial refusal examples，但不要寫成長政策文字。

   訓練 target 應該示範自然的角色內迴避，而不是把 "never explain modern concepts" 這種規則文字塞進資料。例如 user 問 WiFi，assistant 回：「Wi... fie? Dear, if you are fevered, sit by the hearth.」

4. 嘗試 prompt dropout 或 prompt variation。

   不要每筆 sample 都使用完全相同的 system prompt。可以在訓練時變化、縮短或隨機 drop prompt。這可能避免 rote prompt recall，同時保留部分 voice information。

5. 自動化 checkpoint sweep evaluation。

   對每個保存的 adapter checkpoint，產生 traps 並評估三個軸。保留 `in_character`、`refusal`、`helpful`、modern leak count、AI admission count、prompt leak count 的表格。

6. 若 shared adapter 持續混角色，再考慮 per-persona adapters。

   Shared adapter 部署最簡單，仍應優先嘗試。但長輸出中確實有 persona bleed。若 prompt/data 修正後仍無法解決，per-persona adapters 是 fallback。

## 9. 目前狀態

這個專案已成功建立一個完整的本機 RPG NPC SLM 微調實驗室：

- personas 已定義；
- raw 與 curated datasets 已存在；
- MLX-LM LoRA 訓練可運作；
- Qwen 已從 Phi/Gemma 中被選為 base；
- 多種 prompt/data/LoRA variants 已被嘗試；
- 主要 failure modes 已經相當清楚。

但目前還不適合匯出 GGUF 作為最終 NPC brain。下一個有意義的里程碑，是完成 scored M5-v8 evaluation，或針對 v4-v8 checkpoints 做一份統一的 scored comparison。

