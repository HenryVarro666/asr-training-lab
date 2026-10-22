# Controlled Qwen ASR campaign

The supervisor executes a bounded sequence on this host. Source code and passing unit tests are not evidence that the campaign has run. Launch and inspect it with the project virtual environment; all subprocesses use that same Python interpreter.

```bash
cd /mnt/raid/asr-lab/repo
/mnt/raid/asr-lab/venv/bin/python scripts/run_qwen_campaign.py \
  --campaign-dir /mnt/raid/asr-lab/runs/qwen-campaign-NEW \
  --baseline-fixed128 /ABSOLUTE/PATH/TO/fixed128.metrics.json \
  --baseline-dev-clean /ABSOLUTE/PATH/TO/eligible-dev-clean.metrics.json
```

Replace the baseline paths with actual outputs from the pinned original Qwen model. `--baseline-dev-clean` is optional; when omitted, the supervisor generates that baseline. The fixed128 baseline is required before training. Its manifest hash, pinned model revision, utterance coverage, nonempty WER denominator, zero truncation and raw prediction file are verified. Use the shared, speaker-balanced `dev-clean-fixed128.jsonl` produced by `asr_lab.qwen_data`, not the earlier independently sampled Whisper subset.

The campaign directory must be new and empty. A nonblocking lock on its `campaign.lock`, plus the project-level `/mnt/raid/asr-lab/campaign.lock`, prevents concurrent campaign supervisors. Do not launch another experiment manually on the same GPUs while this campaign is active. The supervisor does not inspect, stop or modify unrelated services.

The sequence is:

1. E1: physical GPU1, `qwen_single.yaml`, microbatch 2 and accumulation 16.
2. E2: physical GPUs0,1, `qwen_main.yaml`, microbatch 2 and accumulation 8.
3. E3: physical GPUs0,1, `qwen_batch4.yaml`, microbatch 4 and accumulation 4, when E2 has safe measured headroom.
4. E4: physical GPUs0,1, `qwen_batch8.yaml`, microbatch 8 and accumulation 2, only when E3's maximum per-rank allocated project memory is below 60 GiB and the minimum observed free memory across both physical GPUs exceeds 12 GiB.

Every benchmark starts from the same pinned model, uses global batch32, 2,048 training examples, eight development examples, 40 optimizer updates and ten warmup updates. Steps11–40 supply the 30 measured updates. `asr_lab.benchmark.summarize_run` computes audio-seconds throughput and p50/p95 update time; evaluation, loading and checkpoint writes are outside this metric. Every training loss/gradient norm must be finite. Physical-GPU free-memory evidence comes from the resource monitor; project allocation comes from the trainer's rank-MAX metric.

The fastest valid **dual-GPU** configuration with measured safe headroom starts a fresh full-data one-epoch run at learning rate `1e-5`; no benchmark checkpoint becomes its initialization. The single-GPU result is an efficiency reference and cannot win selection. No second main learning-rate run is launched: its recommendation stays pending analysis of this run's training and evaluation matrix.

The supervisor then validates generation WER on fixed128 and all eligible prepared `dev-clean` rows for both the original model and final checkpoint. If `dev-other.jsonl` is prepared at campaign start, both original and final models also evaluate it. All generation runs use batch16 by default; reused baseline metadata must record the same `evaluation_batch_size`. Reduce it with `--eval-batch-size` only when regenerating matching baselines. Raw predictions and normalized edit-count denominators are retained. This short-form manifest excludes rejected long clips; it is not the complete unfiltered official benchmark. A campaign becomes `complete` only after the full requested evaluation matrix validates.

`campaign.json` contains current state, commands, fingerprints, selection and evaluation results. `experiment_results.json` records each measured, OOM or skipped candidate. `REPORT.md` shows the actual matrix, signed WER differences and outstanding recommendation. Per-stage `stdout.log`, `stage.json`, trainer artifacts and `monitor/` traces remain available. Existing stage directories are reused only with a matching command/config fingerprint and a successful stage receipt; their model or evaluation outputs are still validated. A whole campaign is never silently restarted.

Source and manifest hashes are checked before each stage; a code or data change stops the campaign for review. SIGTERM and Ctrl-C trigger owned-process cleanup.

The global default wall-time budget is 43,200 seconds, including all benchmarks, training and evaluation. It can be reduced with `--max-wall-seconds`; it cannot be increased beyond 12 hours. Remaining time is passed into each trainer. Project storage must stay below 300 GiB and filesystem free space above 500 GiB, accounting for projected checkpoint writes. The main-stage projection uses the measured parameter count: FP32 weights plus two FP32 Adam moments, one GiB minor-state allowance per checkpoint, rounded up, times three for two retained checkpoints and a temporary third. For this model this is about72GiB with the measured 2,038,052,480 unique parameters; existing smoke/benchmark checkpoints are already included in measured project usage and are not added twice. The supervisor does not delete checkpoints.

Each child runs in its own process group, receives a resource monitor sampled every ten seconds and has all HF, Torch, Triton, CUDA, pip and temporary caches placed under `/mnt/raid/asr-lab`. Deadline or interruption cleanup signals only groups created by this supervisor and verifies that those groups have exited. A benchmark CUDA OOM may skip a larger candidate and fall back to an already valid smaller dual candidate; other failures, nonfinite values and data/invariant errors stop the campaign. No user credentials, SSH, external messages or cloud experiment tracking are used.
