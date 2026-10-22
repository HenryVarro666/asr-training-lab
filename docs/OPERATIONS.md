# H100 experiment operations

All project files, environment, downloads, caches, temporary files, checkpoints, and logs belong under `/mnt/raid/asr-lab`. The canonical RAID mount had approximately 3.3 TB free when verified on 2026-10-22; verify current free space before each run. The host root filesystem was nearly full, with less than 1 GiB free. `/data/asr-lab` may exist as a compatibility symlink, but saved manifests and configuration use `/mnt/raid/asr-lab`. Do not install system packages or use root filesystem caches. Preserve existing processes on either GPU, including the observed PersonaPlex process using approximately 22.7 GiB on physical GPU 0; the ASR project does not exclusively own that GPU. The monitor observes every GPU and never kills or alters any training process.

```bash
export TMPDIR=/mnt/raid/asr-lab/tmp
export PIP_CACHE_DIR=/mnt/raid/asr-lab/cache/pip
export XDG_CACHE_HOME=/mnt/raid/asr-lab/cache
export HF_HOME=/mnt/raid/asr-lab/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid/asr-lab/cache/huggingface/datasets
export TORCH_HOME=/mnt/raid/asr-lab/cache/torch
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
source /mnt/raid/asr-lab/venv/bin/activate
```

Use project-specific tmux sessions such as `asr-data`, `asr-train`, `asr-monitor`, and `asr-tensorboard`. `tmux attach -t asr-train` reconnects; detach with Ctrl-b then d. The training launch command, physical GPU reservation, world size, and PID must be saved with run metadata. Verify actual available GPU memory and existing GPU PIDs before launching training; CUDA device indices inside a masked process can differ from physical `nvidia-smi` indices. Do not infer GPU availability from a count of visible devices.

After the train process is launched, run the monitor in its own tmux session. For distributed training, use the long-lived `torchrun` launcher PID, rather than a short-lived shell PID.

```bash
# Substitute the actual run directory and launcher PID:
python scripts/monitor.py --run-dir /mnt/raid/asr-lab/runs/RUN_ID --pid TRAIN_PID --gpu 0,1 --interval 30
scripts/status.sh /mnt/raid/asr-lab/runs/RUN_ID
# A single resource snapshot, useful before launch:
python scripts/monitor.py --run-dir /mnt/raid/asr-lab/runs/RUN_ID --once
```

The monitor writes `monitor/resources.csv`, `monitor/samples.jsonl`, and atomic `monitor/status.json` under the run directory. Samples contain UTC timestamps, training PID state, all GPUs' UUID/utilization/VRAM/temperature/power, host disk free space, last metrics record and age, latest checkpoint path and age, and training terminal status. Run metadata is included from `run.json` (or `run_metadata.json`) when present. The training code owns the separate run-level `status.json` and `summary.json`.

Alerts include nonfinite latest loss, no metrics/stale metrics after15 minutes, no/stale checkpoint after2 hours, free space below 500 GiB, failed GPU queries, and unexpected process exit. A training `status=complete/completed/success` makes process exit normal. Baseline generation can precede the first optimization step; configure `--stale-seconds` for a known long initial evaluation. `scripts/status.sh` adds a `monitor_stale` alert when the last sample is older than90 seconds. Monitor exit does not prove successful training: inspect training status/summary, final dev WER, and checkpoint files.

If space or loss alerts occur, inspect the responsible ASR run and its logs before taking action. Do not kill third-party jobs. Resume only from a saved, complete checkpoint following the training CLI instructions. A `.part` archive or staged extraction is not accepted data. Never infer that training has started merely because data preparation or a smoke test succeeded.

TensorBoard must listen only on loopback:

```bash
tensorboard --logdir /mnt/raid/asr-lab/runs --host 127.0.0.1 --port 6006
# From your own computer, using its configured SSH host alias:
ssh -N -L 6006:127.0.0.1:6006 SSH_HOST_ALIAS
```

Then open `http://127.0.0.1:6006` locally. The actual training TensorBoard event directory follows the run's configuration. Keep all web services bound to `127.0.0.1`; never embed SSH credentials or host secrets in scripts, logs, or this repository.
