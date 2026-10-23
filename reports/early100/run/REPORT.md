# Original-schedule Early100 control

Status: failed; stage: step100-dev_clean.

One fresh initialization; original892-step scheduler and45-step warmup; planned stop after100 updates.
25/50 are model-only exports.100 contains full recovery state, structurally verified; no new resume-training test.
No follow-up training or automatic retry. Full development results cover only eligible audio of at most30 seconds.
Matched aggregate counts/audio do not prove the original per-ID order or bitwise model trajectory.

| Evaluation | N | Base WER | Prefix WER | Difference (percentage points) | S / D / I |
|---|---:|---:|---:|---:|---|
| step25-fixed128 | 128 | 1.5835% | 4.1997% | +2.6162 | 96 / 13 / 13 |
| step50-fixed128 | 128 | 1.5835% | 3.7177% | +2.1343 | 84 / 8 / 16 |
| step100-fixed128 | 128 | 1.5835% | 3.2014% | +1.6179 | 73 / 9 / 11 |

All three preset intermediate points are exploratory diagnostics; no checkpoint is selected as a winner.

Failure: ValueError: Generation truncation prevents an accepted evaluation
