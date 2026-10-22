#!/usr/bin/env bash
set -euo pipefail
CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/mnt/raid/asr-lab/tmp}"
export HF_HOME="${HF_HOME:-/mnt/raid/asr-lab/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/raid/asr-lab/cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/raid/asr-lab/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
exec torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-2}" -m asr_lab.qwen_train "$@"
