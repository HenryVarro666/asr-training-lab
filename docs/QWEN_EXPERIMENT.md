# Qwen3-ASR controlled experiment

Main candidate: official `Qwen/Qwen3-ASR-1.7B`, model revision `7278e1e70fe206f11671096ffdd38061171dd6e5`. The actual loaded model has about 2.349 billion parameters; the model name is retained verbatim. Adapter follows the official Qwen3-ASR SFT API at source commit `7c6daf77a2421100f5fb066495372c00129d39ff`. Sources: [official repository](https://github.com/QwenLM/Qwen3-ASR), [official SFT script](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/qwen3_asr_sft.py), [model](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5). The derived prompt/collator/outer-forward adaptation carries Apache-2.0 attribution; the upstream license text is preserved in `docs/QWEN_UPSTREAM_LICENSE.txt`. Existing Whisper remains a reference experiment; compare models on identical examples.

Use the installed CUDA PyTorch 2.8 environment and `pip install -r requirements-qwen.txt`. This pins qwen-asr 0.0.6, transformers 4.57.6, and accelerate 1.12.0. No FlashAttention compilation or vLLM is required; attention uses SDPA. All caches and temporary files stay under `/mnt/raid/asr-lab`.

```bash
PYTHONPATH=src python -m asr_lab.qwen_data \
  --manifest /mnt/raid/asr-lab/data/manifests/dev-clean.jsonl \
  --output /mnt/raid/asr-lab/data/manifests/dev-clean-fixed128.jsonl --limit 128 --seed 42
PYTHONPATH=src python -m asr_lab.qwen_evaluate \
  --model /mnt/raid/asr-lab/models/QWEN_LOCAL_SNAPSHOT \
  --manifest /mnt/raid/asr-lab/data/manifests/dev-clean-fixed128.jsonl \
  --output /mnt/raid/asr-lab/runs/qwen-baseline/dev-clean-fixed128.predictions.jsonl \
  --batch-size 2 --limit 128
scripts/train_qwen.sh --config configs/qwen_main.yaml \
  --model-path /mnt/raid/asr-lab/models/QWEN_LOCAL_SNAPSHOT \
  --run-dir /mnt/raid/asr-lab/runs/qwen-main
```

The local snapshot path above is a placeholder for the actual pinned downloaded snapshot. If a model ID is supplied, the adapter first resolves the immutable revision to a local snapshot; both model and processor then load from that same snapshot. This avoids the official wrapper's processor revision not following its model revision argument.

Training uses all parameters in FP32 with FP32 gradients and Adam moments; BF16 autocast accelerates matmuls without rounding every small parameter update to BF16. Optimizer is fused AdamW. DDP binds each process to its LOCAL_RANK; no automatic device map. Start with microbatch 2 per GPU and accumulation 8 for effective global batch 32 on two GPUs. `qwen_batch4.yaml` and `qwen_batch8.yaml` preserve global batch 32 using accumulation 4 and 2. A one-GPU control needs accumulation 16. Main learning rate is 1e-5; a separate 5e-6 candidate may be tested after baseline evidence. Default main budget is one epoch with a 12-hour total wall limit. Gradient checkpointing is enabled. Monitor memory headroom alongside throughput and preserve the existing PersonaPlex process on GPU 0.

For short functional/throughput checks use a new run directory with `--max-steps 40 --limit-train 2048 --limit-eval 16`. Time 10 warmup and 30 measured updates from the same base snapshot and identical selected training examples. Do not compare different accumulation counts as if they had the same global batch. Check that both GPUs participate before accepting a distributed result. Real step metrics sum duration/examples over every accumulation microbatch and all ranks; wall time uses the slowest rank and includes data loading between optimizer updates. Evaluation/checkpoint overhead can affect the following update and should be identified in longer comparisons. WER and sample IDs must accompany any quality claim; faster training does not establish better recognition.

The assistant target is `language English<asr_text>TRANSCRIPT` plus EOS. Training explicitly right pads tokenizer and processor and verifies the full token sequence begins with the exact separately tokenized chat/audio prefix. Prefix tokens and padding are ignored in loss using attention masks; EOS is retained even if PAD has the same token ID. Targets stay unshifted because the decoder-only model shifts internally. Generation explicitly left pads, forces the official English prefix, decodes only newly generated tokens, and parses output with the official parser. WER uses the same pinned Whisper EnglishTextNormalizer and spelling map as the Whisper reference, with substitutions/deletions/insertions and truncation counts saved.

The fixed 128-example development manifest is selected deterministically with seed 42 across speakers. Both model baselines must consume this same file; previous baselines on different subsets are preliminary and cannot be directly compared. Online checkpoint evaluation measures dev loss, not generation WER. Run `qwen_evaluate` separately on chosen checkpoints, the full dev-clean set (`--all`), and dev-other before any final quality conclusion. Test-clean remains held out for the selected final model.

Hugging Face checkpoints contain model weights, optimizer/scheduler/Trainer state, and per-rank RNG state. Callback adds the processor plus `checkpoint.json` with config, model/source revisions, manifest hashes, selected train IDs hash, and elapsed wall budget. Retention is two checkpoints. Explicit `--resume /absolute/checkpoint-N` validates the model, data, topology, and config before recovery. `run.json`, `status.json`, `metrics.jsonl`, and TensorBoard events integrate with the existing monitor. Wall-time/SIGTERM stop requests save at an optimizer boundary. No inference web service is launched.

The expanded experiment quota is 300 GiB with at least 500 GiB reserved on the RAID mount. A full Qwen checkpoint includes FP32 weights and Adam state, about 28 GB before minor state files; two retained checkpoints plus a temporary third can require about 84 GB. Space guards budget the next save conservatively; never remove another process's files to make room.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_qwen*.py' -v
```

Loss reduction averages per-microbatch token CE across accumulation steps and DDP ranks. Different microbatch sizes are throughput comparisons; their token weighting is not an exactly identical global-token CE objective. Finite-loss and pre-save disk-guard failures synchronize across ranks before raising.
