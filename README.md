# ASR training lab

Current stage: [saved-matrix CPU numerical repair](reports/token-probability/cpu-repair-20261026-r1/REPORT.md) is complete. The original GPU0 diagnostic completed3 generation +9 teacher-forward calls; its original numeric audit failed and is preserved. FP64 derived statistics now pass the unchanged independent tolerances over all2,304 rows/12 matrices. This CPU phase added **0 model calls and0 training updates**; original130 artifact hashes are unchanged, with monitoring and three figures archived. Stop here; no B stage or new training.

Latest stage: the [four-checkpoint repetition comparison](reports/repetition-onset/README.md) completed in99.16seconds: four singleton calls, identical512-token settings, zero training updates. Base/step25/step50 ended normally with EOS and correctly transcribed this16-word sample; step100 repeated `CULINARY` to512tokens without EOS. Step100 is the earliest observed positive saved checkpoint in this comparison; the exact first update and training cause remain unknown. Raw tokens, EOS, decoder/parser evidence and independent audit are archived. The old Early100 gate remains failed; no next experiment is queued.

Current workflow: the first campaign completed with worse development WER. Follow the [staged diagnosis plan](docs/NEXT_STEPS.md) and [diagnosis evidence](reports/diagnosis/README.md) before any new training. The user has replaced continuous autonomous experimentation with step-by-step planning and review of each stage's evidence. The commands below document the completed campaign workflow; they are not a queued next run. See the [measured results](EXPERIMENT_STATUS.md).

The subsequent [zero-update save/reload control](reports/zero-update-control/README.md) passed on128 fixed inputs: model state, inputs, effective generation rules and raw generated tokens matched exactly, with WER1.5835% on both sides. This narrows the diagnosis without identifying the cause of finetuning regression. The [delivery audit](reports/DELIVERY_AUDIT.md) links the completed training, dual-GPU efficiency, recovery, evaluation and monitoring evidence. The subsequent [Early100 report](reports/early100/README.md) records100 completed dual-H100 updates with the original892-step schedule: fixed128 WER is4.1997%/3.7177%/3.2014% at25/50/100, versus1.5835% originally. Full clean generation hit the512-token limit on one utterance, so its quality gate stopped the campaign; other was not run. This stage is partially completed, not an accepted five-evaluation campaign. See the [archive inventory](reports/ARTIFACT_INVENTORY.md) and [figures](reports/figures/early100/README.md). No further training is queued.

The primary experiment trains the official **Qwen3-ASR-1.7B** architecture on LibriSpeech `train-clean-100` with both H100 GPUs. The loaded model contains about 2.038 billion unique parameters, measured after loading (2,038,052,480), including its acoustic components. Whisper-small remains a separately measured reference. This is supervised finetuning of pretrained ASR models; source code alone does not establish a completed training run or improved WER.

Model weights are pinned to `7278e1e70fe206f11671096ffdd38061171dd6e5`. The prompt, collator and forward adaptation follow official Qwen source `7c6daf77a2421100f5fb066495372c00129d39ff`. See the [Qwen implementation and experiment contract](docs/QWEN_EXPERIMENT.md), [training plan](ASR_TRAINING_PLAN.md), and [upstream license](docs/QWEN_UPSTREAM_LICENSE.txt).

## Run the controlled campaign

Use the existing Python environment under `/mnt/raid/asr-lab/venv`. All data, caches, temporary files, logs and checkpoints live under `/mnt/raid/asr-lab`; preserve other services; the PersonaPlex Demo is paused under explicit user authorization, with its [recovery procedure](reports/token-probability/demo-recovery-20261026/README.md) retained. Dependency versions are in [requirements-qwen.txt](requirements-qwen.txt). Start only after target-host training, checkpoint and resume smoke checks pass.

The [campaign supervisor](docs/CAMPAIGN.md) executes single-GPU and dual-GPU controls at global batch32, measures 30 updates after ten warmup updates, selects a dual-GPU configuration with verified memory headroom, and starts **one fresh full-data epoch** at learning rate `1e-5`. It then validates original/final generation WER on identical fixed128 and complete eligible-development manifests. A second learning-rate experiment stays pending analysis.

Generate the required fixed128 baseline using the public model ID and immutable revision so provenance is recorded. The fixed manifest must already exist:

```bash
cd /mnt/raid/asr-lab/repo
export PYTHONPATH="$PWD/src"
export HF_HOME=/mnt/raid/asr-lab/cache/huggingface
export TMPDIR=/mnt/raid/asr-lab/tmp
export XDG_CACHE_HOME=/mnt/raid/asr-lab/cache
export TORCH_HOME=/mnt/raid/asr-lab/cache/torch
export CUDA_CACHE_PATH=/mnt/raid/asr-lab/cache/cuda
export TRITON_CACHE_DIR=/mnt/raid/asr-lab/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/raid/asr-lab/cache/torchinductor
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
CUDA_VISIBLE_DEVICES=1 /mnt/raid/asr-lab/venv/bin/python -m asr_lab.qwen_evaluate \
  --model Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --manifest /mnt/raid/asr-lab/data/manifests/dev-clean-fixed128.jsonl \
  --output /mnt/raid/asr-lab/runs/qwen-baseline-fixed128/predictions.jsonl \
  --all --batch-size 16
/mnt/raid/asr-lab/venv/bin/python scripts/run_qwen_campaign.py \
  --campaign-dir /mnt/raid/asr-lab/runs/qwen-campaign-NEW \
  --baseline-fixed128 /mnt/raid/asr-lab/runs/qwen-baseline-fixed128/predictions.metrics.json
```

The campaign requires a fresh directory, validates the baseline and manifest hashes, prevents concurrent campaign supervisors, and uses the current Python executable for every subprocess. Its default total wall budget is 12 hours, project quota300GiB and filesystem reserve500GiB. Only process groups it creates are signaled during cleanup. No SSH, credentials, cloud experiment tracker or external publication is required.

## Evidence and evaluation

Each campaign writes `campaign.json`, `experiment_results.json`, `REPORT.md`, per-run training metrics, TensorBoard events, raw generation predictions and resource traces. These artifacts distinguish benchmarks, training, interrupted runs and completed evaluation. Reports show signed WER changes; faster throughput is not evidence of better recognition.

The short-form preparation preserves original FLAC/transcripts, rejects invalid audio and clips above30seconds, and checks split leakage. `dev-clean` currently retains2,694 of2,703 utterances, with nine long clips reported separately. Results on this eligible manifest are not full unfiltered official benchmark results. `dev-other`, when prepared, is evaluated separately; test data remain held out. See [data contracts](docs/DATA.md) and [operations](docs/OPERATIONS.md).

Whisper commands and recovery details are in [the reference guide](docs/WHISPER_REFERENCE.md). Baselines on different128-example subsets cannot be compared directly; use the shared speaker-balanced manifest for model comparisons.

## Verify

```bash
PYTHONPATH=src /mnt/raid/asr-lab/venv/bin/python -m unittest discover -s tests -v
```

Tests exercise target/padding masks, generation-config serialization, nonfinite metric rejection, distributed run creation, checkpoint/RNG recovery, benchmark statistics and campaign controls. Actual CUDA training and resume remain separate integration checks.

Own code uses the MIT license unless a file carries an upstream-derived Apache-2.0 notice. Public model weights and LibriSpeech retain their upstream licenses.
