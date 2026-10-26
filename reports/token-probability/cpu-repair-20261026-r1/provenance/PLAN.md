# Saved-matrix numerical repair: bounded CPU experiment

Status at planning: original A model execution complete; original independent audit failed. This plan precedes the new diagnostic calculations and code changes.

## Scope and immutable inputs

Use only the 12 saved FP32 matrices (2,304 rows, vocabulary 151,936) from `qwen-token-probability-20261026-gpu0-a`, its frozen reference and small tokenizer JSONs. Keep the original run, failed audit, source identities, 512-token generation setting, EOS IDs and raw tokens unchanged. No generation, model construction, weights, GPU, training, B stage or service changes.

New output root: `/mnt/raid/asr-lab/analyses/token-probability-cpu-20261026-r1`; local mirror: `reports/token-probability/cpu-repair-20261026-r1/`. Never overwrite the original failure wrappers or derived results. Save original statistics source and original auditor source plus hashes before editing.

## Budget and stopping conditions

- One focused failure-row diagnosis, at most 120 seconds, including current formula and proposed stable variants against independent FP64.
- Narrow CPU regression tests: at most 120 seconds per validation, at most two validations. Code inspection and review do not consume a GPU budget.
- One full corrected-statistics recomputation, at most 600 seconds; one independent complete audit, at most 600 seconds. No automatic full retry. Combined remote CPU diagnostic/recompute/audit wall budget at most 1,500 seconds including imports and resource monitoring; four CPU threads per process, sequential stages.
- New analysis output at most 128 MiB, process-tree RSS at most 8 GiB; monitor RSS, CPU, elapsed time, phase/row progress and filesystem free space every 5 seconds. Keep filesystem free space above 500 GiB and project below its existing 300 GiB guard. Do not copy matrices, cache weights or install dependencies.
- Stop on source/input hash or byte-size drift, nonfinite values, row/token/EOS mapping changes, resource/deadline violation, or any unchanged audit tolerance failure. Record the failure and partial progress; do not loosen tolerance, rerun a model or extend this plan. Only terminate processes launched by this CPU analysis supervisor.

## Frozen tolerances

`abs_logprob_entropy_margin=5e-5`, `abs_probability=1e-6`, `abs_difference_of_two_logprobs=1e-4`, `nll_sum_error_per_row=5e-5`, `fp64_normalization=2e-12`; ranks, tie counts, argmax and top10 token IDs exact. These are the original audit values.

## Evidence and acceptance

1. Preserve the old failure and source snapshots; reproduce the named failing row without modifying its matrix. Quantify offset/cancellation/normalization effects before choosing a repair.
2. Add an offset/normalization regression and repair only derived CPU arithmetic. Recompute a separate overlay of 12 derived JSONs and three own-trajectory comparisons from original matrices. Record per-file original and new hashes, source identities, and old-versus-new error summaries.
3. Independent auditor must still validate every original worker artifact and immutable input, then verify the corrected overlay against independently implemented FP64 calculations over all 2,304 rows with the same tolerances. Preserve both original failed status and new corrected-analysis status.
4. Only after a full numerical pass, produce corrected comparison summaries/figures and a report stating root cause, maximum errors, coverage, resource receipts and interpretation limits. No new training decision follows automatically.
