#!/bin/bash
# One-shot installer for a stock Vast.ai instance (no Docker image build).
#
# Bootstraps miniforge + the two conda envs, then `pip install -e .` in treegrpo.
# Run from the repo root, after `git clone`:
#
#   bash setup/install.sh
#
# IMPORTANT — the instance MUST be a CUDA **devel** image (nvcc present):
#   `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` is the tested base. flash-attn is
#   compiled from source here, and that needs the CUDA toolkit, not just the driver.
#
# Env vars (all optional):
#   MINIFORGE_DIR        where miniforge is installed
#                        (default /opt/miniforge3 as root, else $HOME/miniforge3)
#   PIP_INDEX_URL        PyPI mirror; unset => official PyPI (pip reads it natively)
#   TORCH_VERSION        default 2.6.0
#   VLLM_VERSION         default 0.8.5.post1
#   TRANSFORMERS_VERSION default 4.53.2
#   FORCE=1              (re)create the envs even if a completion marker exists

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.53.2}"

if [ "$(id -u)" = "0" ]; then
    MINIFORGE_DIR="${MINIFORGE_DIR:-/opt/miniforge3}"
    SUDO=""
else
    MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
    SUDO="sudo"
fi

# Completion markers live outside the repo (repo gets re-cloned/pulled) so a re-run
# can skip the long torch/vllm/flash-attn installs. Delete a marker or FORCE=1 to redo.
MARK_DIR="$HOME/.cache/treegrpo-install"
mkdir -p "$MARK_DIR"

echo "==> [0/6] Checking prerequisites"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "WARNING: nvcc not found. flash-attn will be compiled from source and needs the" >&2
    echo "         CUDA toolkit. If flash-attn install fails below, rent a CUDA *devel* image" >&2
    echo "         (e.g. nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04) and re-run." >&2
fi

echo "==> [1/6] Installing system packages (git, build toolchain)..."
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential ninja-build bzip2

# ---- miniforge ----
if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
    echo "==> [2/6] miniforge already present at $MINIFORGE_DIR"
elif command -v conda >/dev/null 2>&1; then
    MINIFORGE_DIR="$(conda info --base)"
    echo "==> [2/6] using existing conda at $MINIFORGE_DIR"
else
    echo "==> [2/6] Installing miniforge to $MINIFORGE_DIR"
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm -f /tmp/miniforge.sh
fi

CONDA_SH="$MINIFORGE_DIR/etc/profile.d/conda.sh"
source "$CONDA_SH"

# ---- retriever env ----
if [ -f "$MARK_DIR/retriever.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "==> [3/6] retriever env already done (FORCE=1 to rebuild)"
else
    echo "==> [3/6] Creating retriever env (python 3.10.13) + faiss/e5/uvicorn"
    [ -d "$MINIFORGE_DIR/envs/retriever" ] || conda create -y -n retriever python=3.10.13
    conda activate retriever
    pip install --upgrade pip
    pip install torch==${TORCH_VERSION} torchvision==0.21.0 torchaudio==${TORCH_VERSION}
    # faiss-gpu 1.7.3 was yanked from PyPI; the CUDA 12 wheel now starts at 1.8.x. Pin a known-good build.
    pip install "numpy<2"
    pip install faiss-gpu-cu12==1.10.0
    pip install transformers datasets pyserini
    pip install uvicorn fastapi
    conda deactivate
    touch "$MARK_DIR/retriever.done"
fi

# ---- treegrpo env ----
if [ -f "$MARK_DIR/treegrpo.done" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "==> [4/6] treegrpo env already done (FORCE=1 to rebuild)"
else
    echo "==> [4/6] Creating treegrpo env (python 3.12.9) + vllm + flash-attn"
    [ -d "$MINIFORGE_DIR/envs/treegrpo" ] || conda create -y -n treegrpo python=3.12.9
    conda activate treegrpo
    pip install --upgrade pip
    pip install torch==${TORCH_VERSION} torchvision==0.21.0 torchaudio==${TORCH_VERSION}
    pip install vllm==${VLLM_VERSION}
    # transformers >=4.51.1 matches vllm 0.8.5. verl's monkey-patch asserts 4.45-4.47.1, but
    # that patch only runs under ulysses sequence parallelism (multi-GPU); single-GPU skips it.
    pip install "transformers==${TRANSFORMERS_VERSION}" peft pandas
    # Keep datasets/huggingface_hub on the 2.x/0.x line (huggingface_hub <1.0) that transformers requires.
    # Otherwise the later `pip install -e .` pulls latest datasets -> huggingface_hub>=2.0
    # and silently breaks transformers' `<1.0` pin (plus renames the `huggingface-cli`).
    pip install "datasets<3.0" "huggingface_hub>=0.24,<1.0"
    # flash-attn is ABI-coupled to torch: the latest (unpinned) no longer supports torch 2.6.0's
    # cxx11 ABI and loads with `undefined symbol: c10::Error`. Pin 2.7.4.post1 (torch 2.6.0 / vllm 0.8.5).
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    conda deactivate
    touch "$MARK_DIR/treegrpo.done"
fi

# ---- editable install of the repo ----
echo "==> [5/6] pip install -e . (treegrpo env)"
conda activate treegrpo
cd "$WORK_DIR"
# verl's pyproject.toml declares accelerate/tensordict/ray/datasets UNPINNED. A plain
# `pip install -e .` re-resolves them against latest and silently upgrades torch 2.6.0
# (tensordict is ABI-coupled to torch) and datasets/huggingface_hub, breaking vllm/transformers.
# So: install the remaining deps explicitly with the torch-2.6.0-paired tensordict, then
# install verl itself with --no-deps so nothing re-clobbers the pins above.
pip install "tensordict==0.7.0" accelerate codetiming dill hydra-core pybind11 ray
pip install -e . --no-deps
conda deactivate

# ---- conda init ----
echo "==> [6/6] conda init bash"
"$MINIFORGE_DIR/bin/conda" init bash

cat <<EOF

==================== install.sh complete ====================
conda envs:  treegrpo  (python 3.12.9, torch ${TORCH_VERSION}, vllm ${VLLM_VERSION}, flash-attn)
             retriever (python 3.10.13, torch ${TORCH_VERSION}, faiss-gpu-cu12 1.10.0, uvicorn/fastapi)

Open a NEW shell (so conda init takes effect), then:
  source "$CONDA_SH"
  conda activate treegrpo
  bash setup/download_data.sh                    # models + index + parquet

In a separate tmux/terminal:
  conda activate retriever
  bash setup/launch_retriever.sh                 # http://0.0.0.0:8000/retrieve

Back in treegrpo:
  python scripts/make_sft_data.py --teacher models/Qwen2.5-7B-Instruct --out data/sft.parquet
  bash run_sft.sh
  bash run_tree.sh
============================================================
EOF
