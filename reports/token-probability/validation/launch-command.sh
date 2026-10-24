#!/usr/bin/env bash
# One bounded probability-diagnostic campaign; never launch training or retry.
set -u
cd /mnt/raid/asr-lab/repo || exit 90
export PYTHONPATH=/mnt/raid/asr-lab/repo/src
export TMPDIR=/mnt/raid/asr-lab/tmp XDG_CACHE_HOME=/mnt/raid/asr-lab/cache
export HF_HOME=/mnt/raid/asr-lab/cache/huggingface HF_HUB_CACHE=/mnt/raid/asr-lab/cache/huggingface/hub
export TORCH_HOME=/mnt/raid/asr-lab/cache/torch TRITON_CACHE_DIR=/mnt/raid/asr-lab/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/raid/asr-lab/cache/torchinductor CUDA_CACHE_PATH=/mnt/raid/asr-lab/cache/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=''
/mnt/raid/asr-lab/venv/bin/python scripts/run_token_probability.py \
  --run-dir /mnt/raid/asr-lab/runs/qwen-token-probability-20261023-a \
  --max-wall-seconds 1200 \
  > /mnt/raid/asr-lab/runs/qwen-token-probability-20261023-a.stdout.log 2>&1
probe_exit=$?
printf '%s\n' "$probe_exit" > /mnt/raid/asr-lab/runs/qwen-token-probability-20261023-a.exit-code.txt
exit "$probe_exit"
