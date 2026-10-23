# Repetition Onset Probe Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development; preserve the user-selected Astra review / Sol implementation workflow.

**Goal:** Compare original and25/50/100-step weights on the same utterance, preserving512-token generation, to locate the earliest observed saved checkpoint with repetition.

**Architecture:** A new bounded supervisor invokes the existing frozen raw-token worker once per model in separate processes. CPU analysis separates periodic repetition from budget exhaustion and records the limits of four-point temporal localization.

**Tech Stack:** Existing pinned PyTorch/transformers/Qwen environment; Python, JSON, offline local checkpoints.

**Spec:** [REPETITION_ONSET_PROBE.md](../../REPETITION_ONSET_PROBE.md).

## Global Constraints

- Stages base,step025,step050,step100; singleton652-129742-0006; exactly4generate calls/4consumptions/0updates. Only existing specified weights.
- Existing BF16/SDPA/English prompt/leftpadding/greedy/num_beams1/use_cacheTrue/max_new_tokens512; no scores, new processors, stopping criteria or RNG reset.
- Physical GPU1, preserve all other services; total1200seconds, eachworker240seconds, incremental1GiB, project300GiB, free500GiB; offline cache underRAID.
- Freeze existing worker/training/evaluation source and old reports. No retries, other data/models, training, publishing or deleting artifacts.
- Identity/input/effective-rule/runtime drift, OOM, timeout, nonfinite observed inputs, I/O or missing evidence stops remaining calls. Repetition and truncation are observations.

## Review Focus

- Original checkpoint provenance must distinguish historical anchors from newly captured file hashes; never mislabel a live hash as prior proof.
- Changes to tokenizer/config must not silently change effective inference conditions; input/rules checks precede further stages.
- Keep full sequences/EOS/PAD; exclude postEOS padding only in derived repetition statistics.
- Any repeated text/token unit can trigger the predeclared operational metric, not onlyCULINARY. Separate strong repetition from512/noEOS exhaustion.
- Four saved checkpoints do not identify an exact first update or establish monotonicity/training cause.

## Task1: Freeze contract and identity

**Ownership:** Astra writes docs/REPETITION_ONSET_PROBE.md; Sol data ops writes reports/repetition-onset/inputs/model-reference.json and validation/identity-preflight.json.
**Interfaces:** Ordered4 model paths and historical/current source/weight/config receipts; existing target reference and old singleton evidence.

- [x] Save budget, evidence, repetition definitions and stopping rules before GPU work.
- [x] Verify original pinned snapshot and checkpoint25/50/100 files/markers; record tokenizer/config differences and hash provenance honestly.

## Task2: Minimal supervisor and CPU analysis

**Ownership:** Sol training owns scripts/run_repetition_onset.py, src/asr_lab/repetition_metrics.py, tests/test_repetition_onset.py.
**Interfaces:** Consume Task1reference; invoke frozen asr_lab.qwen_truncation_probe with stage target-single; produce independent per-model identities/raw evidence, comparisons, onset summary and bounded campaign state.

- [x] Test fixed4calls/4samples, stop-on-drift/failure, genuine vs short repetition, EOS-padding exclusion, earliest-observed interpretation and nonmonotone patterns.
- [x] Implement within the contract without editing old worker/source.
- [x] Run targeted then fullCPU tests; create review diff package; Astra reviews spec and code quality, resolve blockers and freeze source.
- [x] Parent verifies remote deployment hashes and targetedCPU tests with no visibleCUDA.

## Task3: Execute, audit and archive

**Ownership:** Parent launches/monitors one freshrun and writes report/current status; Sol data ops independently audits saved evidence; Astra reviews interpretation.

- [x] Verify liveGPU1, lock, disk and source; run exactly4preset processes and save actual progress.
- [x] Mirror rawJSON/PT/config/resource traces; independently recompute tokens/EOS/decodes/parser/normalizer/repetition and verify all4input/settings match.
- [x] Report per-checkpoint findings, earliest observed repetition/loop and limitations; retain oldfailed states and verify cleanup/service preservation.
- [ ] Commit allowlisted small artifacts; verify cross-hostGit bundle and receipt, no publicpush or automaticnextstage.

## Execution decisions

The user explicitly authorized plan then execution. Continue in the user-selected local directory and reuse the named Astra/Sol agents; no redundant approval, copying to a worktree, or changes to unrelated untracked from-scratch plans. Use a dedicated ignored ledger; the old experiment contracts remain immutable historical evidence, while this new contract authorizes these four new model comparisons.
