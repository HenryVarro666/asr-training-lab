# Token Probability Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Preserve the user-selected Astra strategy/review and Sol implementation workflow.

**Goal:** Explain the observed conditional-token divergence on one utterance across base/50/100 without training or modifying decoding decisions.

**Architecture:** A bounded supervisor launches three fresh workers. Each captures one official greedy generation with native output-logits/scores observation, then scores the same three frozen token trajectories through the official thinker forward. An independent CPU audit reconstructs probability metrics from full-vocabulary saved matrices.

**Tech Stack:** Existing pinned Qwen/PyTorch/Transformers environment; FP32 stable probability arithmetic; JSON/PT evidence; standard plotting from saved evidence only.

**Spec:** [TOKEN_PROBABILITY_PROBE.md](../../TOKEN_PROBABILITY_PROBE.md).

## Global Constraints

- Only base/step050/step100, audio652-129742-0006, old137-token singleton input and frozen trajectory IDs of lengths27/37/512.
- At most3generate+9explicit teacher-forced forwards,0optimizer/backward calls. Generation remains max512/greedy/beam1/cacheTrue/BF16/SDPA; only output_logits/output_scores observation flags added.
- Total GPU campaign1200s, eachmodel stage240s, newrun/cache2GiB, project300GiB, free500GiB; physicalGPU1, preserve PersonaPlex and other processes. No retry, newdata/model, download or training.
- Full-vocabulary continuation-aligned FP32 logits/scores saved, no duplicate full probability matrices. Old source and reports immutable. Large PT mirrored locally but not copied again into the remote repo or Git.
- CPU post-run audit/derived reporting has600s execution limit and no model/generate calls. Failed audit stops claims; it never authorizes GPU retry.

## Review Focus

- Shift P-1+j must score Y[j], including finalEOS, with input P||Y[:-1] and fresh multimodal positions/cache.
- A native output flag is an explicit observation change; verify actual raw trajectory unchanged beforeTF and stop after preserving evidence if drift.
- Token fragments and differently cased trajectories must retain exactprefix conditioning; a one-token probability is not word/semantic confidence.
- EOS probability sums unique151645/151643 once; raw and processed roles, argmax ties, ranks, underflow and logprob must remain reproducible.
- Cached generation vs no-cache full forward can differ numerically. Save differences and actual score-based decisions, not a fabricated equivalence guarantee.

## Task1: Freeze input and design

**Owners/files:** Astra docs/TOKEN_PROBABILITY_PROBE.md; Sol dataops reports/token-probability/inputs/trajectory-reference.json and validation/host-preflight.json; Sol training inputs/upstream-probability-source.json.
**Interface:** Fixedmodels/trajectories, old raw/input/runtime/effective anchors, current model expectedSHAs from previous captured verification, official generation/forward source provenance.

- [x] Save bounded contract and stop criteria beforeGPU.
- [x] Verify three fixed trajectories and model anchors without model loading or weight scanning.
- [x] Resolve actual official output and fresh-position forward behavior from pinned source.

## Task2: Capture and CPU statistics

**Owner:** Sol training; create src/asr_lab/token_probability.py, scripts/run_token_probability.py, tests/test_token_probability.py. Do not edit frozen old modules.
**Interface:** Newrunner consumes trajectory-reference; worker emits identity/input/runtime, raw-tokens, generation logits/scores, three aligned TF matrices and row mappings, derived per-token candidates/EOS/top10/rank/entropy, actualcall counters and state receipts.

- [x] Test shift includingL1/finalEOS, stable distributions/rank/ties/EOSmass, exactprefix/token preservation, permittedconfig observation flags, argument decisions/scorealignment,3x3bound and fail-stop/no retry.
- [x] Implement newworker and supervisor using existing purehelpers and single current-model SHA/stat identity checks. Override the old1GiB limit for this explicit2GiB contract.
- [x] Run targeted and fullCPU tests; freeze sources; Astra partial review plus independent Sol final review; resolve blockers.
- [x] Deploy exactnewfiles, verifySHAs and focusedremoteCPU tests with no visibleCUDA.

## Task3: Execute and interpret

**Owner:** Parent launches only after review/test/livegates. Sol dataops owns independent audit and evidence-derived figures. Astra reviews final interpretation.

- [ ] FreshGPU1/disk/lock/source check, launch one bounded run, monitor actual phase/calls and preserve every completed/failed artifact.
- [ ] Mirror rawPT+JSON and verify hashes; CPU audit recomputes all rows with independentFP64 reference (predeclared abs logprob/entropy5e-5, probability1e-6; rank/ties/argmax exact), evaluates inputshift/provenance/oldraw/counters, no model calls.
- [ ] Compare exact-prefix branches, correct transcript endings and full512 loop/EOS traces, show numerical-generation/TF differences and limits. Save source-linked PNG/SVG/PDF figures after numeric audit.
- [ ] Update currentreport/status, verify ownedprocess exit/Plex/oldfailedstate, commit allowlisted small artifacts, verified cross-hostbundle plus separate receipts. No automatically queued next stage.

## Execution decision

The user explicitly authorized planning then this finite experiment. Continue in the existing local directory on branchcodex/token-probability. No redundant approval or newthread; preserve unrelated untracked study materials and existing artifacts. No additionalGPU smoke is within this budget.

## Interrupted-run disposition

2026-10-23 resumed: Task2 complete with local109/full and remote9/focusedCPU tests, SHA-bound independent Sol final review following partial Astra review. Task3 resource gate checked twice after resume: GPU1 has foreign processes, intended run absent. No launch, actual3+9 not performed; independent audit/figure helpers prepared but no real rows/figures. State ready_blocked_gpu_busy, no waiting/retry queue. Readiness is not execution completion.
