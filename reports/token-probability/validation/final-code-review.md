# Final independent code review

Status: **passed_ready_not_executed**. Reviewer: independent Sol final reviewer, 2026-10-23 America/Chicago. No remaining blocking finding in the three reviewed files. This is code readiness evidence, not a probability experiment result or GPU execution receipt.

Astra previously reviewed strategy and core implementation but reached its usage limit before final signoff. Astra's review is **partial**, not passed. The parent supplied its remaining deadline finding; this reviewer independently inspected the implementation and the fix. No claim is made that Astra approved the final code.

## Scope and method

Read `docs/TOKEN_PROBABILITY_PROBE.md`, its implementation plan, the frozen input/upstream-source receipts, all three new source/test files, and relevant inherited supervisor, identity, generation-capture, atomic-save and model-verification helpers. Used the review-changes skill and code-review-graph minimal context/change/flow tools first. The graph returned zero indexed changed functions/flows for these untracked new files; its low risk score was not treated as source coverage. Direct source review supplied that coverage.

## Findings resolved and reviewed

- The 240-second stage deadline remains active through post-worker stat checks and complete artifact verification. Validation checks remaining time before work and immediately before/after each artifact hash; expiry prevents the next stage. Deadline checks in final comparison include each output row and the total deadline remains active.
- Teacher inputs are exactly `P || Y[:-1]`; target rows use `P-1+j`, including L=1 and final EOS. Aligned rows are copied into independent CPU FP32 storage, avoiding accidental serialization of prompt logits.
- Official thinker forwarding uses empty KV/cache positions, resets thinker rope deltas for each trajectory, and reuses original audio tensors. Saved pinned official-source excerpts support the API and position-recompute branch. No labels with values, loss/backward/optimizer, training patch, extra smoke inference, or retry path is introduced.
- Native output flags are the only permitted effective generation differences. Raw sequences and distributions precede drift assertions; a changed old raw trajectory stops teacher forwards and subsequent workers. Processed-score argmax uses torch tie behavior. Raw/scores storage sharing follows explicit full equality comparison under the required empty processor list.
- Stable FP32 row statistics retain underflowed-event log probabilities, sum unique EOS IDs, preserve candidate union/rank/ties, and keep full conditioned prefixes reconstructable. Cached/full-forward numerical differences remain observations.
- Existing supervisor owns only its child process groups; GPU1 idleness gates, lock and storage checks remain active. Storage estimates account for exact-equality deduplication; unexpected unequal generation roles are preserved within reservation or fail before exceeding the budget. No new external command construction or unsafe path interpolation was found in the reviewed launch flow.

## Frozen source identities

Implementation owner explicitly confirmed freeze; reviewer re-read hashes afterward.

| File | SHA256 |
|---|---|
| `src/asr_lab/token_probability.py` | `d7deb7ea0d65cb9ef10a6e66ff5c9eca5207068c4397a1fa27a687dfa111c4f6` |
| `scripts/run_token_probability.py` | `e0eff6b1a9560576c69fdb039277e1085a86b7adc1028dfba22b3e1ac5538400` |
| `tests/test_token_probability.py` | `8390ef0c7a118d7f5acbe4c6d5f3e474c4316c1cf737f5b1f951e0b95d47784f` |

## Verification and limits

Reviewer ran with `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`:

- `.venv/bin/python -m unittest discover -s tests -p test_token_probability.py -v`: 8 tests passed during implementation; owner subsequently added and froze a ninth direct hash-deadline test, whose source was also reviewed.
- `.venv/bin/python -m unittest discover -s tests`: 109 tests passed in 4.248 seconds.
- `git diff --check`: passed; untracked new files were reviewed directly.

The first reviewer test invocation omitted `PYTHONPATH=src` and failed module import before any test ran. The corrected command above passed. This was a local test setup error, not a probe attempt. Implementation-owner receipt `local-unit-tests.log` separately records 9 focused and 109 full tests passing; its direct hash-deadline test confirms no second artifact hash starts after expiry.

No model construction, GPU inference, SSH, commit, or actual probe invocation was performed by this reviewer. Actual probe counters remain **0 generate, 0 teacher forward, 0 updates**. CPU mocks/toy tensors and synthetic timing are not model execution or proof of GPU completion within 240 seconds.

The existing timestamped `resume-host.json` reports GPU1 occupied by three other processes and the planned probe run absent. GPU execution is blocked; do not launch or queue automatically. Parent retains responsibility for remote CPU/hash verification and fresh live resource gates before any later explicitly resumed execution. A source change invalidates this SHA-bound signoff. No probability results, scientific interpretation or post-run audit are approved by this review.
