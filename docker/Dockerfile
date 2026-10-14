# Tree-GRPO training image for Vast.ai (single 24GB GPU, Qwen2.5-0.5B).
#
# Two conda envs (matching the upstream README):
#   treegrpo   -> python 3.12.9, torch 2.6.0, vllm 0.8.5.post1, flash-attn, verl (-e .)
#   retriever  -> python 3.10.13, torch 2.6.0, transformers, datasets, pyserini, faiss-gpu, uvicorn, fastapi
#
# The repo is baked in at /workspace/Tree-GRPO; mount your own copy over it if you prefer.
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t youruser/tree-grpo:qwen2.5-0.5b .
#   docker push youruser/tree-grpo:qwen2.5-0.5b

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    MINIFORGE=/opt/miniforge3

# ---- system deps (git + toolchain for building flash-attn) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        build-essential ninja-build \
        bzip2 && \
    rm -rf /var/lib/apt/lists/*

# ---- miniforge ----
RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash /tmp/miniforge.sh -b -p "$MINIFORGE" && \
    rm /tmp/miniforge.sh

ENV PATH="$MINIFORGE/bin:$PATH"
SHELL ["/bin/bash", "-c"]

RUN conda update -y -n base conda

# ============================================================
# retriever env (built first so it layers cheaply and independently)
# ============================================================
RUN conda create -y -n retriever python=3.10.13 && \
    source "$MINIFORGE/etc/profile.d/conda.sh" && conda activate retriever && \
    pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    # faiss-gpu 1.7.3 is not published on PyPI as `faiss-gpu`; use the CUDA 12 wheel.
    pip install "numpy<2" && \
    pip install faiss-gpu-cu12==1.7.3 && \
    pip install transformers datasets pyserini && \
    pip install uvicorn fastapi

# ============================================================
# treegrpo env
# ============================================================
RUN conda create -y -n treegrpo python=3.12.9 && \
    source "$MINIFORGE/etc/profile.d/conda.sh" && conda activate treegrpo && \
    pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    pip install vllm==0.8.5.post1 && \
    # transformers is pinned by verl's monkey-patch to 4.45-4.47.1.
    pip install "transformers==4.47.1" peft pandas && \
    pip install flash-attn --no-build-isolation

# ---- bake in the (possibly locally modified) repo ----
WORKDIR /workspace/Tree-GRPO
COPY . /workspace/Tree-GRPO

RUN source "$MINIFORGE/etc/profile.d/conda.sh" && conda activate treegrpo && \
    pip install -e .

# Convenience: `conda activate <env>` works in interactive shells.
RUN conda init bash

# Make the launcher scripts executable.
RUN chmod +x /workspace/Tree-GRPO/*.sh /workspace/Tree-GRPO/setup/*.sh 2>/dev/null || true

WORKDIR /workspace/Tree-GRPO
CMD ["/bin/bash"]
