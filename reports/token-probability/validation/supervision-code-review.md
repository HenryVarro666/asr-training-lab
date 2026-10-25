# Independent A-supervision code review

Status: **passed_ready_not_executed**, GPT-6.1 Sol independent reviewer probability_review,2026-10-25.

The reviewer used graph-first discovery (empty graph required direct-source fallback), inspected the A-only changes, and independently ran four absolute-boundary CPU regressions,4/4 OK in0.428 seconds. bash -n and git diff --check passed. No SSH, GPU, model load or training by the reviewer.

Resolved blocking findings:
- Initial candidate only accounted for heavy imports without an active deadline. Final stdlib entry preparse arms SIGALRM before scientific imports; slow-import timeout writes a zero-call partial receipt without overwriting existing run.
- Initial candidate wrote complete before finalization. Final code persists models_complete_pending_cleanup and commits complete campaign.json last, after cleanup/terminal observations/report writes. General finalize failure and denied complete+failure writes cannot leave durable complete. Primary budget_stopped classification survives finalization errors.

Other reviewed requirements: actual versus planned invocations, present versus size/SHA-verified matrices, unknown counts remain unknown; immediate and ten-second stdlib preflight observations; owned-process-only cleanup within absolute cap; fresh paths/symlinks/locks; selected GPU UUID/visibility gates and PersonaPlex rejection; no retry/fallback/B/training.

Scientific worker differences are limited to selected resource checks and the explicit runtime.physical_gpu resource-identity exception. Other runtime/source/input gates, old raw trajectory equality,3generations+9TF,512tokens,1200total/240stage seconds,2GiB, and statistical definitions remain unchanged.

SHA-bound reviewed source:
- scripts/run_token_probability.py:82f7f95d91b7ae39044899cf12870c3cbc9aa58723baab3a59a118e4e10f7b43
- src/asr_lab/probe_supervision.py:2b6db25983f1d7a81feb902f793dbadb436b6afd4b1e5c166c64785256b8723b
- src/asr_lab/token_probability.py:acc34e309468c950b02e930ee7be7d45a0394b092ad0b0a35e07f83016f3a43b
- supervision-launch.sh:f99cec787abb3699d7f36d3296bddbf7f7c7f0c9d3c657b26872ea8f64cfb8d8

No runtime result is approved by this review. Parent must deploy exact files, complete remote CPU validation and refresh GPU0/1/lock/disk/source gates before a single fresh run.

## Final v2 delta review (supersedes runner/test SHA above)

Remote CPU validation exposed a genuine pre-cleanup failure-publication gap in v1. The slow-tree regression saw running rather than budget_stopped under a0.4s Linux budget. Failed logs/deployment are preserved under v1-failed names. The new source commits primary failure plus cleanup_pending atomically before expensive teardown, with unknown invocation counts for launched unvalidated workers; the absolute deadline is not reset. The tiny timer regression now isolates external sampler/cleanup latency; a separate ten-second real fake-resource integration still covers the sampler.

Independent probability_review scoped review PASS,1/1 new durable-failure regression in0.405s, diff check passed. Final SHA runner ea10ddfe07cbc40b751dc088d0b3927d7f1e09ac0a828d4ca720932140b16c6e; test_token_probability.py6a6427c0d37171d6c3a53871b500a78169c3019712fd289c437bafbec278df46. Helper/worker/launcher signatures unchanged. Local final25+6 focused /131 full CPU tests pass; remote rerun remains required. No A or model call has yet been made in this new attempt.
