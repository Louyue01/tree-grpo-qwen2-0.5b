#!/bin/bash
# Download all models + retrieval index/corpus, then build the parquet datasets.
# Run this inside the `treegrpo` conda env (torch/transformers/datasets/huggingface_hub installed).
#
# Env vars (with defaults):
#   WORK_DIR    repo root (default: parent of this script)
#   MODEL_DIR   where HF models are saved (default $WORK_DIR/models)
#   DATA_DIR    where train/test/sft parquet are written (default $WORK_DIR/data)
#   INDEX_DIR   where the wiki-18 FAISS index + corpus live (default $WORK_DIR/retrieval)
#   BASE_MODEL  base model HF id (default Qwen/Qwen2.5-0.5B-Instruct)
#   TEACHER     teacher model HF id for SFT demo generation (default Qwen/Qwen2.5-7B-Instruct)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$WORK_DIR/models}"
DATA_DIR="${DATA_DIR:-$WORK_DIR/data}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/retrieval}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "$MODEL_DIR" "$DATA_DIR" "$INDEX_DIR"

# Download an HF repo to a local dir, working across huggingface_hub CLI versions:
# `huggingface-cli` (old) / `hf` (new) / plain python snapshot_download (always works).
hf_download() {
    local repo="$1" dir="$2"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo" --local-dir "$dir"
    elif command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$dir"
    else
        python - "$repo" "$dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
    fi
}

echo "==> [1/4] Downloading base model: $BASE_MODEL"
if [ ! -d "$MODEL_DIR/$(basename "$BASE_MODEL")" ]; then
    hf_download "$BASE_MODEL" "$MODEL_DIR/$(basename "$BASE_MODEL")"
fi

echo "==> [2/4] Downloading teacher model for SFT demos: $TEACHER"
if [ ! -d "$MODEL_DIR/$(basename "$TEACHER")" ]; then
    hf_download "$TEACHER" "$MODEL_DIR/$(basename "$TEACHER")"
fi

# Align the base tokenizer's ChatML chat_template with the teacher's. Base Qwen2.5 models
# ship WITHOUT a chat_template (and rl_dataset.py only ChatML-wraps the prompt when
# tokenizer.chat_template is set), while the multi-hop reward (qa_em_format.is_valid_sequence)
# requires the resulting <|im_start|>assistant marker. Instruct already has the template, but
# copying the teacher's guarantees the SFT demos and RL use the exact same wrapper either way.
echo "==> Copying ChatML chat_template from teacher to base model"
python "$WORK_DIR/scripts/add_chat_template.py" \
    --base "$MODEL_DIR/$(basename "$BASE_MODEL")" \
    --teacher "$MODEL_DIR/$(basename "$TEACHER")"

echo "==> [3/4] Downloading e5-base-v2 retriever encoder"
if [ ! -d "$MODEL_DIR/e5-base-v2" ]; then
    hf_download intfloat/e5-base-v2 "$MODEL_DIR/e5-base-v2"
fi

echo "==> [4/4] Downloading wiki-18 FAISS index + corpus"
if [ ! -f "$INDEX_DIR/e5_Flat.index" ]; then
    python "$WORK_DIR/scripts/download.py" --save_path "$INDEX_DIR"
    cat "$INDEX_DIR"/part_* > "$INDEX_DIR/e5_Flat.index"
    rm -f "$INDEX_DIR"/part_*
fi
if [ ! -f "$INDEX_DIR/wiki-18.jsonl" ]; then
    gzip -dk "$INDEX_DIR/wiki-18.jsonl.gz"
fi

echo ""
echo "==> Processing multi-hop QA datasets into parquet"
FLASHRAG_DIR="${FLASHRAG_DIR:-RUC-NLPIR/FlashRAG_datasets}" \
    WORK_DIR="$WORK_DIR" LOCAL_DIR="$DATA_DIR" \
    bash "$WORK_DIR/scripts/data_process/data_process_multihop.sh"

echo ""
echo "Done. Layout:"
echo "  models/      base + teacher + e5 encoder"
echo "  retrieval/   e5_Flat.index + wiki-18.jsonl"
echo "  data/        train.parquet + test.parquet"
echo ""
echo "Next: generate SFT demos (with the retriever running) then run SFT warmup:"
echo "  python scripts/make_sft_data.py --teacher $MODEL_DIR/$(basename "$TEACHER") --out $DATA_DIR/sft.parquet"
echo "  bash run_sft.sh"
