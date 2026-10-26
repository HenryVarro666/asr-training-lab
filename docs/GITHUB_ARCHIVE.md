# GitHub archive and complete experiment evidence

User authorized this GitHub export on2026-10-26. Repository: [HenryVarro666/train](https://github.com/HenryVarro666/train). It was an empty **public** repository before this initial push; its visibility is preserved. The export uses the existing local Git history, without rewriting commits.

## Latest verified experiment

- Source/results payload: commit `9f5e95c0536bf2476f306de14147ff6c58f0cba6`.
- [CPU numerical repair report](../reports/token-probability/cpu-repair-20261026-r1/REPORT.md), [machine summary](../reports/token-probability/cpu-repair-20261026-r1/summary.json), [unchanged-tolerance independent audit](../reports/token-probability/cpu-repair-20261026-r1/overlay-independent-audit.json).
- All2,304 rows/12 matrices pass after FP64 derived-statistics repair. The original audit failure is preserved separately; no model rerun, training or B stage followed.
- Three figures each have PNG/SVG/PDF exports and source SHA records. Fifty focused CPU tests passed.
- All130 original files, totaling1,462,218,349 bytes, retain their SHA256 and sizes on both hosts.

## Storage contract

| Artifact | GitHub | Complete local/RAID copy |
|---|---|---|
| Code, configuration, plans, reports, progress, small receipts and SHA manifests | Yes | Yes |
| Original failure, original/corrected statistics-source snapshots and independent audit | Yes | Yes |
| PNG/SVG/PDF figures and figure-source identities | Yes | Yes |
| Full raw model distribution matrices, model weights, audio and datasets | No | Yes |
| Large per-token derived JSON and full cross-model comparison JSON | SHA manifest and summaries | Yes |

A GitHub clone does not contain the complete raw or large derived evidence. Links from reports to intentionally excluded derived JSON refer to the local/RAID mirror. Use the committed manifests to verify any separately transferred payload.

- Original local mirror: `reports/token-probability/attempts/qwen-token-probability-20261026-gpu0-a/`.
- Corrected local mirror: `reports/token-probability/cpu-repair-20261026-r1/`.
- Original server root: `/mnt/raid/asr-lab/runs/qwen-token-probability-20261026-gpu0-a`.
- Corrected server root: `/mnt/raid/asr-lab/analyses/token-probability-cpu-20261026-r1`.
- [Corrected manifest](../reports/token-probability/cpu-repair-20261026-r1/recomputed/manifest.json) binds12derived and3own-comparison files; [archive inventory](../reports/token-probability/cpu-repair-20261026-r1/archive-inventory.json) binds the full mirror.

GitHub `main` is the published latest snapshot; `codex/token-probability` preserves the active working branch. No unrelated untracked learning/course drafts are part of this export. A small push receipt outside Git records the verified remote refs, so it does not recursively change the commit being verified.
