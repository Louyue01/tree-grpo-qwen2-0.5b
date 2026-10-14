#!/bin/bash
# Launch the local dense (e5 + FAISS) retrieval server on http://0.0.0.0:8000/retrieve.
# The RL rollout and SFT demo generation POST to this endpoint.
#
# Run inside (or let this script activate) the `retriever` conda env, which has
# faiss-gpu + transformers + datasets + uvicorn + fastapi.
#
# Env vars (with defaults):
#   WORK_DIR       repo root (default: parent of this script)
#   MODEL_DIR      dir holding e5-base-v2 (default $WORK_DIR/models)
#   INDEX_DIR      dir holding e5_Flat.index + wiki-18.jsonl (default $WORK_DIR/retrieval)
#   TOP_K          number of passages per query (default 3)
#   RETRIEVER_ENV  conda env name (default retriever); set AUTO_ACTIVATE=1 to activate it here

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$WORK_DIR/models}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/retrieval}"
TOP_K="${TOP_K:-3}"
RETRIEVER_ENV="${RETRIEVER_ENV:-retriever}"

if [ "${AUTO_ACTIVATE:-0}" = "1" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$RETRIEVER_ENV"
fi

INDEX_FILE="$INDEX_DIR/e5_Flat.index"
CORPUS_FILE="$INDEX_DIR/wiki-18.jsonl"
RETRIEVER_MODEL="$MODEL_DIR/e5-base-v2"

for f in "$INDEX_FILE" "$CORPUS_FILE" "$RETRIEVER_MODEL"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: missing $f — run setup/download_data.sh first." >&2
        exit 1
    fi
done

echo "Launching retriever (topk=$TOP_K) on http://0.0.0.0:8000/retrieve"
echo "  index:  $INDEX_FILE"
echo "  corpus: $CORPUS_FILE"
echo "  model:  $RETRIEVER_MODEL"

python "$WORK_DIR/search_r1/search/retrieval_server.py" \
    --index_path "$INDEX_FILE" \
    --corpus_path "$CORPUS_FILE" \
    --topk "$TOP_K" \
    --retriever_name e5 \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
