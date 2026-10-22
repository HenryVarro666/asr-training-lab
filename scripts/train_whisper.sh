#!/usr/bin/env bash
set -euo pipefail
ROOT="${ASR_ROOT:-/mnt/raid/asr-lab}"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR"
cd "$ROOT/repo"
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Set CUDA_VISIBLE_DEVICES=1 for a single-GPU smoke. Launch with 0,1 only after
# checking the existing GPU0 service has adequate remaining memory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" -m asr_lab.train "$@"
