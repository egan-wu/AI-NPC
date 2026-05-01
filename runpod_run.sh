#!/bin/bash
# runpod_run.sh — 全自動上傳 → 安裝 → 訓練 → 生成 eval → 下載
#
# Usage:
#   bash runpod_run.sh <HOST> <PORT>
#
# Example:
#   bash runpod_run.sh 157.157.221.29 33962

set -e

# ── 必要參數 ─────────────────────────────────────────────────────────────────
HOST="${1:?請提供 HOST，例如: bash runpod_run.sh 157.157.221.29 33962}"
PORT="${2:?請提供 PORT}"
KEY="${3:-$HOME/.ssh/id_ed25519}"   # 預設 id_ed25519，可覆寫

# ── 固定路徑（不需改動）──────────────────────────────────────────────────────
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="~/slm"
SSH_OPTS="-i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-i $KEY -P $PORT -o StrictHostKeyChecking=no"
TARGET="root@$HOST"
RUN_NAME="p2c_mistral7b_best"
RESPONSES_JSON="outputs/eval/${RUN_NAME}_responses.json"

echo "================================================"
echo " RunPod 自動訓練腳本"
echo " Host : $HOST:$PORT"
echo " Key  : $KEY"
echo " Local: $LOCAL_ROOT"
echo "================================================"
echo ""

# ── 1. 建立遠端目錄結構 ───────────────────────────────────────────────────────
echo "[1/5] 建立遠端目錄..."
ssh $SSH_OPTS $TARGET "mkdir -p \
    $REMOTE_DIR/data/mlx_curated_medium \
    $REMOTE_DIR/outputs/adapters \
    $REMOTE_DIR/outputs/eval \
    $REMOTE_DIR/configs \
    $REMOTE_DIR/scripts"

# ── 2. 上傳檔案 ───────────────────────────────────────────────────────────────
echo "[2/5] 上傳檔案..."
scp $SCP_OPTS \
    "$LOCAL_ROOT/data/mlx_curated_medium/train.jsonl" \
    "$LOCAL_ROOT/data/mlx_curated_medium/valid.jsonl" \
    "$TARGET:$REMOTE_DIR/data/mlx_curated_medium/"

scp $SCP_OPTS \
    "$LOCAL_ROOT/configs/training_p2c_mistral7b.yaml" \
    "$LOCAL_ROOT/configs/personas_medium.yaml" \
    "$TARGET:$REMOTE_DIR/configs/"

scp $SCP_OPTS \
    "$LOCAL_ROOT/scripts/05_train_hf.py" \
    "$LOCAL_ROOT/scripts/06_generate_responses_hf.py" \
    "$TARGET:$REMOTE_DIR/scripts/"

scp $SCP_OPTS \
    "$LOCAL_ROOT/setup_runpod.sh" \
    "$TARGET:$REMOTE_DIR/"

echo "   上傳完成"

# ── 3. 安裝環境 ───────────────────────────────────────────────────────────────
echo "[3/5] 安裝環境（約 5 分鐘）..."
ssh $SSH_OPTS $TARGET "cd $REMOTE_DIR && bash setup_runpod.sh"

# ── 4. 訓練 ──────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] 開始訓練（約 15–30 分鐘，輸出即時顯示）..."
ssh $SSH_OPTS $TARGET "cd $REMOTE_DIR && python scripts/05_train_hf.py --config configs/training_p2c_mistral7b.yaml"

# ── 5. 生成 eval 回應 ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] 生成 eval 回應..."
ssh $SSH_OPTS $TARGET "cd $REMOTE_DIR && python scripts/06_generate_responses_hf.py \
    --model    mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter  outputs/adapters/p2c_mistral7b/best_adapter \
    --personas configs/personas_medium.yaml \
    --run-name $RUN_NAME \
    --output   $RESPONSES_JSON"

# ── 6. 下載結果 ───────────────────────────────────────────────────────────────
echo ""
echo "[6/6] 下載 responses JSON..."
mkdir -p "$LOCAL_ROOT/outputs/eval"
scp $SCP_OPTS \
    "$TARGET:$REMOTE_DIR/$RESPONSES_JSON" \
    "$LOCAL_ROOT/$RESPONSES_JSON"

echo ""
echo "================================================"
echo " 完成！"
echo " 結果已存至: $LOCAL_ROOT/$RESPONSES_JSON"
echo ""
echo " 下一步：在 Claude session 裡讀取並打分"
echo " 完成後記得到 RunPod 網頁停止 Pod 節省費用"
echo "================================================"
