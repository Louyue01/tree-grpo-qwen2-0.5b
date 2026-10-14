#!/bin/bash
# Convenience wrapper: launch Tree-GRPO (tree search) RL training.
# All configuration lives in train_multihopqa_tree_search.sh (see its header for env vars).
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$BASE_DIR/train_multihopqa_tree_search.sh" "$@"
