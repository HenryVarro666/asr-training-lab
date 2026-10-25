#!/usr/bin/env bash
# One reviewed stage-A attempt, selected physical card; no retry or training.
set -u
if [[ $# -ne 2 ]]; then exit 64; fi
probe_run_dir=$1
probe_physical_gpu=$2
case "$probe_run_dir" in /mnt/raid/asr-lab/runs/qwen-token-probability-*) ;; *) exit 64 ;; esac
case "$probe_physical_gpu" in 0|1) ;; *) exit 64 ;; esac
if [[ -e "$probe_run_dir" || -L "$probe_run_dir" || -e "$probe_run_dir.stdout.log" || -e "$probe_run_dir.exit-code.txt" ]]; then exit 73; fi
cd /mnt/raid/asr-lab/repo || exit 90
export PYTHONPATH=/mnt/raid/asr-lab/repo/src
export TMPDIR=/mnt/raid/asr-lab/tmp XDG_CACHE_HOME=/mnt/raid/asr-lab/cache
export HF_HOME=/mnt/raid/asr-lab/cache/huggingface HF_HUB_CACHE=/mnt/raid/asr-lab/cache/huggingface/hub
export TORCH_HOME=/mnt/raid/asr-lab/cache/torch TRITON_CACHE_DIR=/mnt/raid/asr-lab/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/raid/asr-lab/cache/torchinductor CUDA_CACHE_PATH=/mnt/raid/asr-lab/cache/cuda
export PIP_CACHE_DIR=/mnt/raid/asr-lab/cache/pip
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=''
/mnt/raid/asr-lab/venv/bin/python scripts/run_token_probability.py \
  --run-dir "$probe_run_dir" --physical-gpu "$probe_physical_gpu" --max-wall-seconds 1200 \
  > "$probe_run_dir.stdout.log" 2>&1
probe_exit=$?
printf '%s\n' "$probe_exit" > "$probe_run_dir.exit-code.txt"
exit "$probe_exit"
