# Data and training coverage receipts

These small receipts were copied from `/mnt/raid/asr-lab/data` after the completed campaign. No audio, archives, or model weights are included in this directory.

- `archives.json`: preparation-time official MD5 checks, local SHA256, byte lengths, download URLs and verification timestamp for LibriSpeech train-clean-100, dev-clean and dev-other. This is the original receipt, not a claim that the archives were re-downloaded or rehashed during the later diagnosis.
- `manifest-report.json`: decoded-PCM hash definition, ID/speaker/audio leakage checks, accepted/rejected counts, hours and manifest hashes. Training contains 28,539 accepted clips / 100.59088 hours; clean accepts2,694 and rejects9 long clips; other accepts2,857 and rejects7 long clips.
- `training-coverage.json`: recomputed from the892 training-step JSONL records. The run consumed28,544 examples / 100.60493 audio hours. The five additional consumptions are consistent with two-rank batch padding; exact per-example exposure IDs were not logged. Do not describe the epoch as every unique clip being consumed exactly once.

The common manifest and raw-reference checks for the completed Whisper-small full eligible clean reference are recorded in `../baselines/whisper-full-clean/verification.json`. The recovered TensorBoard event file was parsed into `../monitoring/tensorboard-verification.json`, which confirms892 loss/gradient/lr records and8 development-loss evaluations.
