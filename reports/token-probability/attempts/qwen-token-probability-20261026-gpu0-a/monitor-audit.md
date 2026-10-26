# Independent lifecycle and resource audit

Status: **passed monitor semantics**. Only inventory-SHA-matched small receipts were read; no PT/model/SSH access.

Campaign514.432s<1200s; three sequential GPU0 workers exit0;3 generate+9 explicit TF,0 updates. cleanup_pending=false; no owned processes remain. All sampled GPU0 compute PIDs belong to the workers and all observations use the same GPU0 UUID. Terminal GPU0 has no compute PID.

| Stage | Worker seconds | Whole-stage conservative upper bound seconds |
|---|---:|---:|
| base | 53.272 | 160.134 |
| step050 | 62.337 | 199.967 |
| step100 | 95.917 | 164.815 |

Monitor52samples: first9.503s in initial_resource_observation before every weight SHA; interval min/median/max4.815/9.998/10.140s; terminal stopped/complete at514.428s. No monitor errors.

All9 explicit TF receipts are now inventory-SHA-verified after complete mirror. Each is complete, inference-only, labels=None, use_cache=false, past_key_values/cache_position/position_ids=None, rope reset=None, no returned cache. Saved JSON IDs equal P || Y[:-1]; masks and audio tensor fingerprints are preserved; mapped positions are136+j with exact frozen targets and27/37/512 full-vocabulary rows. Correct EOS remains the last predicted target; loop has no added EOS. No PT was read.

- First observation at9.503s after entry precedes tree traversal by signed source ordering; no independently timestamped tree-start event is saved.
- Whole-stage conservative bounds bracket observed phase transitions, including SHA/worker/validation; worker durations alone exclude SHA.
- Resource samples about10s apart cannot establish an absolute peak or exclude brief subinterval external GPU activity.
- Legacy per-worker monitor fields say training/checkpoint and may end running/zombie; these names are generic monitor artifacts, not evidence of training. Final preflight stopped receipt and owned cleanup are authoritative.
- No standalone lifecycle event log; campaign/stage/metadata receipts jointly provide lifecycle evidence.
- No PT file read; complete mirror SHA and independent FP64 numerical audit remain separate.

Closure: numerical audit remains failed_no_retry (exit1,15.597s; original tolerance unchanged), with0 result figures. This small-receipt closure is not a numerical audit pass.
