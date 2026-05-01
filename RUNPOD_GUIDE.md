# RunPod 訓練流程 — P2-C Mistral 7B

## 概覽

```
本地 Mac                           RunPod Pod
──────────────────                 ──────────────────────────────
1. 建立 Pod（網頁）        →
2. 複製 SSH 指令           →       pod 啟動
3. scp 上傳檔案            →       收到 dataset + scripts
                                   4. bash setup_runpod.sh
                                   5. python 05_train_hf.py
                                   6. python 06_generate_responses_hf.py
7. scp 下載 responses.json ←
8. 放入 outputs/eval/，打分         （在本地 session 裡）
9. 停止 Pod（省錢）
```

---

## 步驟 1：建立 Pod

1. 登入 [runpod.io](https://www.runpod.io) → **Deploy**
2. 選 **GPU**（推薦排序）：
   - RTX 4090 24GB（最快，~$0.74/hr）
   - RTX 3090 24GB（便宜，~$0.34/hr）
   - A100 40GB（若上面沒貨）
3. Template：選 **`RunPod PyTorch 2.1`**（內建 CUDA，不用自裝）
4. Disk：**20 GB** container disk（模型 ~14GB + adapter + dataset）
5. 點 **Deploy**，等 pod 狀態變 **Running**

---

## 步驟 2：取得 SSH 連線

Pod 頁面 → **Connect** → 複製 SSH 指令，格式如：
```
ssh root@<IP> -p <PORT> -i ~/.ssh/id_rsa
```

如果還沒設定 SSH key：
- RunPod 網頁 → Settings → SSH Public Keys → 貼上你的 `~/.ssh/id_rsa.pub`

---

## 步驟 3：從本地上傳檔案

在**本地 Mac** 執行（把 `<IP>` 和 `<PORT>` 換成你的 pod 資訊）：

```bash
# 切到專案根目錄
cd /Users/egan-wu/Documents/workspace/claude_workspace/small-language-model-world

# 在 pod 上建立目錄結構
ssh root@<IP> -p <PORT> "mkdir -p ~/slm/data/mlx_curated_medium ~/slm/outputs/adapters ~/slm/outputs/eval ~/slm/configs ~/slm/scripts"

# 上傳 dataset
scp -P <PORT> data/mlx_curated_medium/train.jsonl  root@<IP>:~/slm/data/mlx_curated_medium/
scp -P <PORT> data/mlx_curated_medium/valid.jsonl   root@<IP>:~/slm/data/mlx_curated_medium/

# 上傳 configs
scp -P <PORT> configs/training_p2c_mistral7b.yaml   root@<IP>:~/slm/configs/
scp -P <PORT> configs/personas_medium.yaml           root@<IP>:~/slm/configs/

# 上傳 scripts
scp -P <PORT> scripts/05_train_hf.py                root@<IP>:~/slm/scripts/
scp -P <PORT> scripts/06_generate_responses_hf.py   root@<IP>:~/slm/scripts/

# 上傳 setup script
scp -P <PORT> setup_runpod.sh                       root@<IP>:~/slm/
```

---

## 步驟 4：SSH 進去，安裝環境

```bash
ssh root@<IP> -p <PORT>

cd ~/slm
bash setup_runpod.sh
```

安裝大約 3–5 分鐘。最後會印出 GPU 名稱和 VRAM 確認。

---

## 步驟 5：訓練

```bash
cd ~/slm

# 用 tmux 跑，斷線不影響（建議）
tmux new -s train

python scripts/05_train_hf.py --config configs/training_p2c_mistral7b.yaml
```

**預計時間**：
- RTX 3090：約 20–30 分鐘（150 steps）
- RTX 4090：約 12–18 分鐘

訓練中會每 25 steps 印一次 val loss。完成後印出完整 val loss 表，
並儲存最佳 adapter 到 `outputs/adapters/p2c_mistral7b/best_adapter/`。

---

## 步驟 6：生成 eval 回應

```bash
# 在同一個 tmux session 或新開一個
python scripts/06_generate_responses_hf.py \
    --model    mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter  outputs/adapters/p2c_mistral7b/best_adapter \
    --personas configs/personas_medium.yaml \
    --run-name p2c_mistral7b_best \
    --output   outputs/eval/p2c_mistral7b_responses.json
```

約 3–5 分鐘，生成 30 條回應並儲存 JSON。

如果想評估特定 checkpoint（例如 step-75），adapter 路徑改成：
```
outputs/adapters/p2c_mistral7b/checkpoint-75
```

---

## 步驟 7：下載回應到本地

在**本地 Mac** 執行：

```bash
cd /Users/egan-wu/Documents/workspace/claude_workspace/small-language-model-world

scp -P <PORT> root@<IP>:~/slm/outputs/eval/p2c_mistral7b_responses.json \
    outputs/eval/p2c_mistral7b_responses.json
```

然後在 Claude session 裡讀取並打分。

---

## 步驟 8：停止 Pod（重要！省錢）

訓練完成、檔案下載後，立刻停止 pod：
- RunPod 網頁 → Pod → **Stop**（保留磁碟，下次可繼續）
- 或 **Terminate**（完全刪除，最省）

---

## 費用估算

| 操作 | 時間 | RTX 3090 費用 |
|---|---|---|
| setup_runpod.sh | ~5 min | ~$0.03 |
| 訓練 150 steps | ~25 min | ~$0.14 |
| eval 生成 | ~5 min | ~$0.03 |
| **合計** | **~35 min** | **~$0.20** |

---

## 常見問題

**CUDA OOM（記憶體不足）**：
- 把 `per_device_train_batch_size` 從 4 改成 2
- 或 `gradient_accumulation_steps` 從 1 改成 2（效果等同 batch_size=4）

**Unsloth 安裝失敗**：
- 試 `pip install unsloth` 不帶額外選項

**tmux 操作**：
- 離開（保持執行）：`Ctrl+B` 然後 `D`
- 回到 session：`tmux attach -t train`
- 查看所有 session：`tmux ls`
