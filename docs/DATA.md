# LibriSpeech data contract

The initial experiment uses `train-clean-100` for training and `dev-clean` for model selection. `test-clean` is optional and remains held out until a final selected checkpoint is evaluated. Development WER is not a held-out test result. Full 1,000 h training and audiobook source archives are not downloaded.

Official source: [OpenSLR SLR12](https://www.openslr.org/12). LibriSpeech contains 16 kHz read English speech derived from LibriVox audiobooks, prepared by Vassil Panayotov with Daniel Povey. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve the extracted `LibriSpeech/LICENSE.TXT` and other root metadata when redistributing. Citation: Panayotov, Chen, Povey, and Khudanpur, “LibriSpeech: an ASR corpus based on public domain audio books,” ICASSP 2015. This project filters and manifests the original utterances without editing speech or transcripts.

Official [checksum list](https://www.openslr.org/resources/12/md5sum.txt), verified 2026-10-22:

| Archive | MD5 |
| --- | --- |
| train-clean-100.tar.gz | 2a93770f6d5c6c964bc36631d331a522 |
| dev-clean.tar.gz | 42e2234ba48799c1f50f24a7926300a1 |
| test-clean.tar.gz | 32fa31d27d2e1cad72775fee3f4849a9 |

```bash
# In the checkout, with its installed Python environment active:
export TMPDIR=/mnt/raid/asr-lab/tmp
scripts/download_data.sh --root /mnt/raid/asr-lab/data --project-root /mnt/raid/asr-lab
# Optional final held-out split, using the same validation:
scripts/download_data.sh --root /mnt/raid/asr-lab/data --splits test-clean
# After separate official archive downloads into raw/*.tar.gz.part:
PYTHONPATH=src python -m asr_lab.prepare_data prepare --root /mnt/raid/asr-lab/data
```

Downloads use HTTPS, resume `.tar.gz.part` files with curl, and finalize only after the pinned official MD5 matches. SHA256 of the archive is also recorded. A changed or incomplete archive is preserved for inspection and fails preparation. Default parallelism is 2, bounded to 4. Interrupted extraction stays uncommitted; successful extraction is staged, rejects absolute paths, traversal, links, and special files, and saves a completion marker. Never delete a server directory to recover automatically from a checksum/extraction failure.

The default guard retains at least 500 GiB filesystem free space and keeps `/mnt/raid/asr-lab` below 300 GiB, including project caches and runs. It conservatively budgets 24 GiB for initial train100 acquisition/extraction and 2 GiB per small split. Guards check before preparation and periodically while downloading. Run directories and temporary/cache files must remain below `/mnt/raid/asr-lab`; these guards do not govern unrelated host jobs.

Artifacts:

- `/mnt/raid/asr-lab/data/raw/{split}.tar.gz`: verified official archive; `.part` means incomplete/unverified.
- `/mnt/raid/asr-lab/data/raw/LibriSpeech/{split}/`: extracted FLAC/transcripts.
- `/mnt/raid/asr-lab/data/archives.json`: official URL, archive MD5/SHA256, bytes, verification time.
- `/mnt/raid/asr-lab/data/manifests/{split}.jsonl`: accepted utterances, deterministic ID order.
- `/mnt/raid/asr-lab/data/manifests/{split}.rejected.jsonl`: every filtered/missing/invalid item and reason.
- `/mnt/raid/asr-lab/data/manifests/report.json`: accepted/rejected counts, retained hours/speakers, manifest hashes, filter and hash definitions for all prepared splits (prior split reports are preserved). Re-run preparation after moving the corpus so absolute audio paths and manifest hashes reflect the canonical location.

Each accepted JSONL row has `id`, absolute `audio_filepath`, original `text`, `duration` in seconds, `sample_rate`, `split`, `source`, `speaker_id`, and `sha256`. The hash is SHA256 of decoded mono 16 kHz IEEE float32 little-endian PCM, excluding FLAC metadata. SoundFile decodes the entire utterance to check finite samples, nonempty/nonnull audio, mono audio,16kHz, and duration `0 < duration <= 30 s`. Longer utterances and all invalid inputs are recorded in the rejection report; do not describe the filtered set as exactly 100 h. All audio paths are local.

Preparation refuses cross-split overlap of utterance IDs, speaker IDs, or exact decoded PCM, including previously prepared optional held-out manifests. It does not remove common transcript sentences: legitimate identical text from different recordings is retained. Within-split duplicate IDs are rejected and reported. Model code must use only the designated train manifest for gradient updates and dev for selection; adding test data to train is prohibited by this experiment contract.

Smoke verification without corpus downloads:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_prepare_data.py -v
```
