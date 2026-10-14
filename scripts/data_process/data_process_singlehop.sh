#!/bin/bash
# Process the single-hop QA datasets into the parquet format expected by the RL trainer.
# Train: NQ train split. Test: NQ / TriviaQA / PopQA.
#
# Env vars (with defaults):
#   WORK_DIR      - repo root (default: parent of this script)
#   LOCAL_DIR     - where train.parquet / test.parquet are written
#   FLASHRAG_DIR  - HF repo id or local dir holding the FlashRAG datasets (default: RUC-NLPIR/FlashRAG_datasets)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_DIR="${LOCAL_DIR:-$WORK_DIR/data}"
FLASHRAG_DIR="${FLASHRAG_DIR:-RUC-NLPIR/FlashRAG_datasets}"

mkdir -p "$LOCAL_DIR"

## single-hop train set (NQ train)
python "$WORK_DIR/scripts/data_process/qa_search_train_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources nq \
    --data_dir "$FLASHRAG_DIR"

## single-hop test set (3 benchmarks)
python "$WORK_DIR/scripts/data_process/qa_search_test_merge.py" \
    --local_dir "$LOCAL_DIR" \
    --data_sources nq,triviaqa,popqa \
    --data_dir "$FLASHRAG_DIR"
