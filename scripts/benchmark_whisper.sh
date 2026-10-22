#!/usr/bin/env bash
set -euo pipefail
ROOT="${ASR_ROOT:-/mnt/raid/asr-lab}"
cd "$ROOT/repo"
source "$ROOT/venv/bin/activate"
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export HF_HOME="$ROOT/cache/huggingface" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
BASE="${1:-$ROOT/runs/whisper-scaling-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$BASE" ]]; then
  echo "Refusing to overwrite benchmark directory: $BASE" >&2
  exit 1
fi
mkdir -p "$BASE"
export ASR_BENCHMARK_BASE="$BASE"
python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['ASR_BENCHMARK_BASE'])
base = yaml.safe_load(Path('configs/whisper_small_librispeech100.yaml').read_text())
# Match global batch32 and schedule; only GPU count/accumulation differ.
for count in (1, 2):
    config = dict(base, batch_size=4, gradient_accumulation_steps=8 // count,
                  eval_steps=1000, save_steps=1000, eval_limit=8, warmup_steps=20)
    (root / f'{count}gpu.yaml').write_text(yaml.safe_dump(config))
PY
# These are sequential to keep each benchmark's GPU allocation explicit.
CUDA_VISIBLE_DEVICES=1 NPROC_PER_NODE=1 scripts/train_whisper.sh \
  --config "$BASE/1gpu.yaml" --run-dir "$BASE/1gpu" --max-steps 70
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 scripts/train_whisper.sh \
  --config "$BASE/2gpu.yaml" --run-dir "$BASE/2gpu" --max-steps 70
PYTHONPATH=src python -m asr_lab.benchmark --one-gpu "$BASE/1gpu" --two-gpu "$BASE/2gpu" \
  --warmup-steps 20 --measure-steps 50 --output "$BASE/summary.json"
