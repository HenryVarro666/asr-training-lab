# Truncation Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute and review each task. Preserve the user's already-authorized Astra strategy / Sol implementation workflow.

**Goal:** Locate the generation, decoding and parsing behavior of utterance652-129742-0006 while retaining checkpoint100 and its512-token generation budget, with no training.

**Architecture:** An isolated inference worker mirrors the frozen evaluator and captures evidence at each boundary. A bounded supervisor runs exactly three independent processes, then compares the saved evidence on CPU. Existing training/evaluation code and historical failure receipts remain unchanged.

**Tech Stack:** Existing pinned PyTorch2.8.0+cu128, transformers4.57.6, qwen-asr0.0.6; Python/JSON; existing campaign process/monitor utilities.

**Spec:** [TRUNCATION_PROBE.md](../../TRUNCATION_PROBE.md).

## Global Constraints

- Only Early100 checkpoint100, physical GPU1; preserve GPU0 PersonaPlex.
- Exact original sorted dev-clean batch137 (indices2192:2208, target index11) twice, then target alone once; three independent workers,3generate calls,33sample consumptions,16unique IDs.
- BF16/SDPA, eval/inference, left padding, English prefix, greedy,num_beams1,use_cacheTrue,max_new_tokens512. No optimizer or gradients; no additional generation kwargs, trace processors or stopping rules.
- Total1200seconds; per worker240seconds; incremental1GiB; project300GiB; disk reserve500GiB. Offline existing model and tokenizer caches only.
- Expected truncation is an observation. Identity drift, timeout, OOM, I/O/worker failure or incomplete mandatory evidence stops remaining workers with no retries.
- No other model, full-dev/other evaluation, training, publishing, deletion or mutation of prior experiment outputs.

## Review Focus

- Batch recreation must use sorted dataset order, not unsorted manifest lines; verify source hashes and exact16IDs.
- Shared padded continuation width must not be mistaken for every row's generated length; preserve all EOS/PAD positions and the full returned sequences.
- Trace collection must not change kwargs, scores, RNG, processor order or stop conditions; compare both raw and effective config.
- Decode and parser observations must derive from the saved same tokens; official repetition cleanup must be visible rather than mistaken for model output.
- Three fresh processes skip the original preceding137batches; non-reproduction cannot falsify the historical event, and single-row success cannot certify a padding bug or fix.

## Task1: Frozen inputs and experiment contract

**Files:** docs/TRUNCATION_PROBE.md; reports/truncation-probe/inputs/reference-batch.json; upstream-generation-source.json; upstream-parser-source.json.
**Interfaces:** exact16rows/order and historical predictions, expected checkpoint/manifest/source hashes; official function source with full-file SHA256.

- [x] Astra defines finite design, budget and stop conditions before GPU work.
- [x] Sol data ops extracts the16ID reference and CPU-verifies paired references/counts/hashes.
- [x] Parent captures exact installed official generate/parser and repetition-helper source without loading GPU weights.

## Task2: Instrumented worker and bounded supervisor

**Files:** Create src/asr_lab/qwen_truncation_probe.py, scripts/run_truncation_probe.py, tests/test_truncation_probe.py.
**Interfaces:** Consume Task1reference, original manifest and checkpoint100completion marker; produce per-worker input metadata, sequences/continuation token IDs, EOS/PAD records, both decodes, parser output, status and stage receipt, plus CPU comparison and campaign status.

- [x] Add meaningful CPU contracts: exact batch slice/order; same512kwargs; prefix slicing; EOS/PAD preservation including late EOS; raw/token-to-decode/parser boundaries; strict3calls/33samples; budget/identity rejection; stop remaining stages on worker error.
- [x] Implement independent inference-only worker; save raw tokens before decode/parsing and preserve partial failure evidence.
- [x] Implement supervisor with global lock, owned process groups, offline cache paths, per-stage/total budgets and resource monitoring; expected truncation does not rewrite old evaluation acceptance.
- [x] Run focused then full CPU suite once; retain logs. Astra independently reviews implementation against contract; resolve blocking findings before GPU launch.
- [x] Parent commits reviewed source and verifies local/remote deployment hashes.

## Task3: Real bounded execution and analysis

**Files:** reports/truncation-probe/run/, README.md, validation/; current project entry/status docs; Git bundle receipt.
**Interfaces:** Use a fresh remote /mnt/raid/asr-lab/runs/qwen-truncation-probe-20261023-* directory; small artifacts mirror locally. Never upload or download large weights/audio as outputs.

- [x] Capture fresh GPU/process/disk/source preflight and verify checkpoint/manifest/audio identity within the total run budget.
- [x] Launch the single authorized supervisor; monitor actual stage progress and only its owned processes.
- [x] Sync raw evidence and compare same-batch inputs, effective settings, tokens/EOS, decoded text and parser output; distinguish repeat/single and historical comparisons.
- [x] Record observed mechanism and limitations; review results with Astra, update progress/entry docs, verify host cleanup and service preservation.
- [x] Commit small evidence and verify a cross-host code/report bundle; finish the finite diagnostic without training or a new queue.

## Execution rulings

The user explicitly requested planning followed by execution, so no redundant confirmation is added. Reuse the already-selected Astra/Sol agents and the user-selected local directory on experiment/truncation-probe; do not copy the entire project to another worktree. Unrelated untracked from-scratch plans/specs are outside scope. Keep a dedicated ignored SDD ledger for this plan; historical Early100 files are evidence, not authority to widen scope.
