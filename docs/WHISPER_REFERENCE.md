# Whisper ASR reference

The primary architecture track is Qwen3-ASR-1.7B. Its implementation and validation are being added separately; the runnable Whisper module below remains the established reference and DDP efficiency probe. Full one-epoch Whisper finetuning is optional while the Qwen track is brought up.

Full parameter finetuning of the unchanged `openai/whisper-small` model (244M parameters) on LibriSpeech `train-clean-100`. The first stage is a bounded ASR experiment, with generation WER measured before training and throughout finetuning on a fixed, seeded `dev-clean` subset. Pretrained baseline, fine-tuned checkpoints, and later full-development evaluation are distinct results. No training result is claimed by this source repository alone.

The configuration pins the public model revision. This code does not upload models, use cloud experiment trackers, or require Hugging Face credentials. Model weights and LibriSpeech retain their upstream licenses; MIT covers this repository's own code.

## Setup

Use Python 3.10–3.13. On the training host, keep dependencies, downloads, temporary files and caches on `/mnt/raid/asr-lab`. Install the CUDA-compatible PyTorch wheel first, then:

```bash
/mnt/raid/asr-lab/venv/bin/python -m pip install -e /mnt/raid/asr-lab/repo
```

Prepare verified FLAC manifests with `python -m asr_lab.prepare_data --help` and the accompanying download scripts. The training inputs are `/mnt/raid/asr-lab/data/manifests/train-clean-100.jsonl` and `dev-clean.jsonl`. Each row carries `id`, absolute `audio_filepath`, `text`, `duration`, `sample_rate`, `split`, `source`, `speaker_id`, and `sha256`. Metadata and decoded audio must be 16 kHz mono. The data audit excludes clips outside `(0, 30]` seconds and labels exceeding Whisper's 448-token target context; it never silently truncates training audio or text.

## Run

Start with the otherwise-free GPU1 and a separate smoke run directory:

```bash
CUDA_VISIBLE_DEVICES=1 NPROC_PER_NODE=1 scripts/train_whisper.sh \
  --config configs/whisper_small_librispeech100.yaml \
  --run-dir /mnt/raid/asr-lab/runs/smoke-001 \
  --max-steps 2 --limit-train 64 --limit-eval 8
```

The wrapper configures all runtime caches under `/mnt/raid/asr-lab`. Explicitly inspect GPU memory and the existing GPU0 service before a two-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 scripts/train_whisper.sh \
  --config configs/whisper_small_librispeech100.yaml \
  --run-dir /mnt/raid/asr-lab/runs/whisper-small-ls100-001
```

At two GPUs, batch 4 per device × accumulation 4 yields a global batch of 32, except the last partial update. One epoch uses AdamW, bf16 autocast, SDPA, learning rate `1e-5`, 5% linear warmup, weight decay `.01`, and gradient norm clipping at 1. The optimization objective averages microbatch mean token losses across accumulation steps and DDP ranks; the reported loss separately weights all valid tokens globally. The run metadata records sampler drops and selected/consumed epoch coverage. The first optimizer update uses the warmup scheduler's zero initial rate. DataLoader workers prepare 30-second log-mel inputs on CPU from lazy FLAC reads; no precomputed feature cache is required.

Before the first update, rank zero runs greedy generation on a fixed seeded sample of at most 128 development utterances, sampled from sorted IDs. The chosen IDs and rejected records are saved. Exactly the same sample is evaluated every 200 optimizer steps and at each epoch/run endpoint. Each prediction preserves raw and normalized reference/hypothesis. WER uses Whisper's English text normalizer and jiwer's corpus word edit counts. Generation has a 440-token budget and reports any budget exhaustion. Test-clean is reserved for a later explicit final evaluation.

## Artifacts and recovery

`run.json` records the config, pinned/resolved model revision, package versions, git revision/dirty state, input manifest hashes, seeds, DDP topology, and effective batch. `metrics.jsonl` and TensorBoard contain token-weighted loss, normalized WER, gradient norm, learning rate, timing, batch audio seconds, throughput, and peak CUDA memory. `predictions/` contains baseline and subsequent generation results. `train_data_audit.json` and `eval_data_audit.json` retain selection and exclusion evidence.

Each atomic checkpoint contains standard Hugging Face model/processor files plus optimizer, scheduler, per-rank Python/NumPy/Torch/CUDA RNG state and the distributed sampler's next epoch/batch cursor. `checkpoints/last` and `checkpoints/best` are atomic symlinks; at most three checkpoint directories are retained. `best` selects among fine-tuned development measurements, independently of the pretrained baseline.

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 scripts/train_whisper.sh \
  --config configs/whisper_small_librispeech100.yaml \
  --run-dir /mnt/raid/asr-lab/runs/whisper-small-ls100-001 \
  --resume /mnt/raid/asr-lab/runs/whisper-small-ls100-001/checkpoints/last
```

Resume requires unchanged data hashes, topology, batch/accumulation, schedule, seed, and selection limits. Repeat the original `--max-steps`/subset arguments for a smoke-run resume. Checkpoints occur every 100 completed optimizer updates and after each development evaluation. A 43,200-second cumulative wall-time budget stops with `budget_stopped`, distinct from completed training. A resumed run reconstructs the epoch order, skips already-consumed batches, then restores model RNG. Fixed seeds/order do not promise bit-identical CUDA kernels. SIGINT/SIGTERM requests a coordinated checkpoint at the next update boundary without waiting for an extra development evaluation; do not use SIGKILL when graceful recovery is wanted.

Evaluate all eligible development clips separately (no `--limit`; this still filters clips above 30 seconds and overlong labels):

```bash
python -m asr_lab.evaluate \
  --model /mnt/raid/asr-lab/runs/whisper-small-ls100-001/checkpoints/best \
  --manifest /mnt/raid/asr-lab/data/manifests/dev-clean.jsonl \
  --output /mnt/raid/asr-lab/runs/whisper-small-ls100-001/full-dev-predictions.jsonl
```

The original dev-clean has 2,703 utterances; the prepared short-form manifest retains 2,694 and reports nine clips above 30 seconds. This filtered evaluation is not the full standard LibriSpeech benchmark. Long-form handling must be added before claiming full official dev-clean WER.

For a fair comparison on the eligible development subset, repeat this command with the original model ID and pinned `--revision` and a different output path. A small-subset improvement is not complete filtered-development or test-set evidence.

## Tests

```bash
python -m unittest discover -s tests -v
```

CPU contracts cover EOS/padding masking, normalized WER, optimizer-update cursors and resume invariants, along with the separate data-preparation tests. A real GPU smoke run and checkpoint-resume probe remain necessary to validate the training integration on the target host.

## Whisper scaling probe

`scripts/benchmark_whisper.sh [new-output-directory]` runs 70 optimizer steps first on GPU1, then on both GPUs. Both runs keep global batch 32 and identical 20-step learning-rate warmup. The summary uses steps 21–70, reports p50/p95 step time, summed-audio/summed-step-time throughput, two-GPU speedup and scaling efficiency. Generation evaluation, checkpoint I/O and model loading are outside these update timings. The probe is an experiment to execute, not evidence that two GPUs are faster.

CUDA runs request fused AdamW; CPU runs record the explicit unfused fallback. DDP gradients use bucket views; static-graph assumptions are not enabled. See [PyTorch 2.8 AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html) and [DDP](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html).

## Controlled Qwen campaign

The sequential campaign supervisor is documented in [CAMPAIGN.md](CAMPAIGN.md). It benchmarks fixed-global-batch configurations, selects a measured safe dual-GPU configuration, starts one fresh full-data epoch and validates original/final generation WER. It requires an existing fixed128 baseline with the pinned model revision and the same evaluation batch size (default16). A 12-hour campaign budget and storage guards bound the work. Commands, resource traces, raw predictions and signed WER differences are retained; no improvement or completed run is inferred from the source alone.
