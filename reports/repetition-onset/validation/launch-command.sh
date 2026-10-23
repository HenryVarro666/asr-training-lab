#!/usr/bin/env bash
# Fixed four-call inference-only probe; no retry or training continuation.
set -u
cd /mnt/raid/asr-lab/repo || exit 90
export PYTHONPATH=/mnt/raid/asr-lab/repo/src
export TMPDIR=/mnt/raid/asr-lab/tmp
export XDG_CACHE_HOME=/mnt/raid/asr-lab/cache
export HF_HOME=/mnt/raid/asr-lab/cache/huggingface
export HF_HUB_CACHE=/mnt/raid/asr-lab/cache/huggingface/hub
export TORCH_HOME=/mnt/raid/asr-lab/cache/torch
export TRITON_CACHE_DIR=/mnt/raid/asr-lab/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/raid/asr-lab/cache/torchinductor
export CUDA_CACHE_PATH=/mnt/raid/asr-lab/cache/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=''
/mnt/raid/asr-lab/venv/bin/python scripts/run_repetition_onset.py \
  --run-dir /mnt/raid/asr-lab/runs/qwen-repetition-onset-20261023-a \
  --model-reference reports/repetition-onset/inputs/model-reference.json \
  --max-wall-seconds 1200 \
  > /mnt/raid/asr-lab/runs/qwen-repetition-onset-20261023-a.stdout.log 2>&1
probe_exit=$?
printf '%s\n' "$probe_exit" > /mnt/raid/asr-lab/runs/qwen-repetition-onset-20261023-a.exit-code.txt
exit "$probe_exit"
