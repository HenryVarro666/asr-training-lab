"""Verified LibriSpeech download and reproducible, leakage-checked manifests."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

SOURCE = "https://www.openslr.org/12"
BASE_URL = "https://www.openslr.org/resources/12"
CHECKSUMS = {
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
}
GIB = 1024**3


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tree_size(root: Path) -> int:
    total = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()]
        for name in files:
            path = Path(parent) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def check_space(project: Path, reserve_gib: float = 500, quota_gib: float = 300,
                projected_bytes: int = 0) -> dict:
    free = shutil.disk_usage(project).free
    used = tree_size(project)
    if free - projected_bytes < reserve_gib * GIB:
        raise RuntimeError(f"Space guard: {free/GIB:.1f} GiB free; must retain {reserve_gib} GiB after projected writes")
    if used + projected_bytes > quota_gib * GIB:
        raise RuntimeError(f"Project quota: {used/GIB:.1f} GiB used + {projected_bytes/GIB:.1f} GiB projected exceeds {quota_gib} GiB")
    return {"free_bytes": free, "project_bytes": used}


def digest_file(path: Path, algorithm="md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(split: str, raw: Path, project: Path, reserve: float, quota: float,
                 base_url: str = BASE_URL) -> dict:
    target = raw / f"{split}.tar.gz"
    partial = raw / f"{split}.tar.gz.part"
    expected = CHECKSUMS[split]
    if target.exists():
        if digest_file(target) != expected:
            raise RuntimeError(f"Checksum mismatch in existing {target}; preserve and investigate")
    else:
        # An independently downloaded complete .part can be finalized without a network call.
        complete = partial.exists() and digest_file(partial) == expected
        if not complete:
            command = ["curl", "--fail", "--location", "--continue-at", "-", "--retry", "4",
                       "--retry-delay", "3", "--connect-timeout", "30", "--output", str(partial),
                       f"{base_url}/{split}.tar.gz"]
            process = subprocess.Popen(command)
            try:
                while process.poll() is None:
                    check_space(project, reserve, quota)
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f"curl failed for {split}: exit {process.returncode}; .part retained")
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        actual = digest_file(partial)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch {partial}: expected {expected}, got {actual}; .part retained")
        os.replace(partial, target)
    return {"split": split, "url": f"{base_url}/{split}.tar.gz", "path": str(target),
            "md5": expected, "sha256": digest_file(target, "sha256"), "bytes": target.stat().st_size}


def safe_extract(archive: Path, raw: Path, split: str) -> Path:
    destination = raw / "LibriSpeech" / split
    marker = raw / f".{split}.extracted.json"
    if destination.exists():
        if marker.exists() and json.loads(marker.read_text()).get("md5") == CHECKSUMS[split]:
            return destination
        raise RuntimeError(f"Unverified existing extraction {destination}; preserve and investigate")
    stage = Path(tempfile.mkdtemp(prefix=f".extract-{split}-", dir=raw))
    try:
        with tarfile.open(archive, "r:gz") as corpus:
            members = corpus.getmembers()
            for member in members:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "LibriSpeech":
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f"Unsafe archive member type: {member.name}")
            for member in members:
                path = Path(member.name)
                # Archives also carry root LICENSE, README, SPEAKERS, BOOKS, CHAPTERS.
                if len(path.parts) > 2 and path.parts[1] != split:
                    raise ValueError(f"Unexpected split in archive: {member.name}")
                target = stage / path
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = corpus.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        staged_split = stage / "LibriSpeech" / split
        if not staged_split.is_dir():
            raise ValueError(f"Archive contains no {split}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_split, destination)
        for metadata in (stage / "LibriSpeech").iterdir():
            if metadata.is_file() and not (destination.parent / metadata.name).exists():
                shutil.copyfile(metadata, destination.parent / metadata.name)
        atomic_json(marker, {"md5": CHECKSUMS[split], "archive": str(archive)})
        return destination
    finally:
        shutil.rmtree(stage)


def build_records(split_dir: Path, split: str, max_duration: float = 30.0):
    import soundfile as sf
    import numpy as np
    records, rejects = [], []
    ids = set()
    transcript_paths = sorted(split_dir.rglob("*.trans.txt"))
    if not transcript_paths:
        raise RuntimeError(f"No transcripts in {split_dir}")
    referenced = set()
    for transcript in transcript_paths:
        for number, line in enumerate(transcript.read_text().splitlines(), 1):
            parts = line.strip().split(maxsplit=1)
            utterance = parts[0] if parts else f"line-{number}"
            path = transcript.parent / f"{utterance}.flac"
            referenced.add(path)
            record = {"id": utterance, "audio_filepath": str(path.resolve()), "split": split}
            try:
                if len(parts) != 2 or not parts[1].strip():
                    raise ValueError("empty_or_malformed_transcript")
                if utterance in ids:
                    raise ValueError("duplicate_utterance_id")
                ids.add(utterance)
                samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
                if sample_rate != 16000:
                    raise ValueError(f"sample_rate_{sample_rate}")
                if samples.shape[1] != 1:
                    raise ValueError("not_mono")
                if not samples.size or not np.isfinite(samples).all():
                    raise ValueError("empty_or_nonfinite_audio")
                duration = samples.shape[0] / sample_rate
                if not math.isfinite(duration) or duration <= 0 or duration > max_duration:
                    raise ValueError(f"duration_out_of_range_{duration:.6f}")
                if not np.any(samples):
                    raise ValueError("silent_audio")
                # Canonical little-endian float32 decoded samples, insensitive to FLAC metadata.
                pcm_hash = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
                record.update(text=parts[1].strip(), duration=duration, sample_rate=sample_rate,
                              source="LibriSpeech/OpenSLR12", speaker_id=utterance.split("-")[0],
                              sha256=pcm_hash)
                records.append(record)
            except (OSError, ValueError, RuntimeError) as error:
                rejects.append({**record, "reason": str(error), "transcript": str(transcript), "line": number})
    for orphan in sorted(set(split_dir.rglob("*.flac")) - referenced):
        rejects.append({"id": orphan.stem, "audio_filepath": str(orphan.resolve()),
                        "split": split, "reason": "audio_without_transcript"})
    return records, rejects


def ensure_disjoint(datasets: dict[str, list[dict]]) -> None:
    for field in ("id", "speaker_id", "sha256"):
        owners = {}
        for split, records in datasets.items():
            for record in records:
                value = record[field]
                previous = owners.setdefault(value, split)
                if previous != split:
                    raise ValueError(f"Cross-split leakage: {field}={value} in {previous} and {split}")


def write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare", "all"), nargs="?", default="all")
    parser.add_argument("--root", type=Path, default=Path("/mnt/raid/asr-lab/data"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=list(CHECKSUMS), default=["train-clean-100", "dev-clean"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=500)
    parser.add_argument("--quota-gib", type=float, default=300)
    parser.add_argument("--max-duration", type=float, default=30)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        parser.error("workers must be between1 and4")
    root = args.root.resolve()
    project = (args.project_root or root.parent).resolve()
    if not root.is_relative_to(project):
        parser.error("data root must be inside project root for quota enforcement")
    raw, manifests = root / "raw", root / "manifests"
    raw.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Conservative allowance covers archive + extracted FLAC + PCM decode working set.
        projected = sum(24 * GIB if s == "train-clean-100" else 2 * GIB for s in args.splits
                        if not (raw / "LibriSpeech" / s).exists())
        check_space(project, args.reserve_gib, args.quota_gib, projected)
        if args.action in ("download", "all"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download_one, split, raw, project, args.reserve_gib,
                                       args.quota_gib, args.base_url) for split in args.splits]
                archives = [future.result() for future in futures]
            atomic_json(root / "archives.json", {"retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "source": SOURCE, "checksum_source": BASE_URL + "/md5sum.txt", "archives": archives})
        if args.action == "download":
            return
        datasets, rejections, verified_archives = {}, {}, []
        for split in args.splits:
            archive = raw / f"{split}.tar.gz"
            if not archive.exists() and (raw / f"{split}.tar.gz.part").exists():
                partial = raw / f"{split}.tar.gz.part"
                if digest_file(partial) == CHECKSUMS[split]:
                    os.replace(partial, archive)
            if not archive.exists() or digest_file(archive) != CHECKSUMS[split]:
                raise RuntimeError(f"Missing or unverified archive {archive}; run download first")
            verified_archives.append({"split": split, "url": f"{args.base_url}/{split}.tar.gz",
                "path": str(archive), "md5": CHECKSUMS[split],
                "sha256": digest_file(archive, "sha256"), "bytes": archive.stat().st_size})
            check_space(project, args.reserve_gib, args.quota_gib)
            directory = safe_extract(archive, raw, split)
            datasets[split], rejections[split] = build_records(directory, split, args.max_duration)
            if not datasets[split]:
                raise RuntimeError(f"No accepted records for {split}")
        prior_archives = {}
        if (root / "archives.json").exists():
            prior_archives = {item["split"]: item for item in json.loads((root / "archives.json").read_text()).get("archives", [])}
        prior_archives.update({item["split"]: item for item in verified_archives})
        atomic_json(root / "archives.json", {"verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": SOURCE, "checksum_source": BASE_URL + "/md5sum.txt",
            "archives": list(prior_archives.values())})
        # Include previously prepared held-out splits when only one new split is requested.
        leakage_sets = dict(datasets)
        for path in manifests.glob("*.jsonl"):
            if path.stem in CHECKSUMS and path.stem not in datasets:
                leakage_sets[path.stem] = [json.loads(line) for line in path.read_text().splitlines()]
        ensure_disjoint(leakage_sets)
        previous_split_reports = {}
        if (manifests / "report.json").exists():
            previous_split_reports = json.loads((manifests / "report.json").read_text()).get("splits", {})
        report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "source": SOURCE,
                  "license": "CC BY 4.0", "max_duration": args.max_duration,
                  "sha256_definition": "decoded mono float32 little-endian PCM,16kHz",
                  "leakage_checks": ["utterance_id", "speaker_id", "decoded_pcm_sha256"], "splits": previous_split_reports}
        for split, records in datasets.items():
            records.sort(key=lambda record: record["id"])
            write_jsonl(manifests / f"{split}.jsonl", records)
            write_jsonl(manifests / f"{split}.rejected.jsonl", rejections[split])
            report["splits"][split] = {"accepted": len(records), "rejected": len(rejections[split]),
                "max_duration": args.max_duration, "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "hours": sum(r["duration"] for r in records) / 3600,
                "speakers": len({r["speaker_id"] for r in records}),
                "manifest_sha256": digest_file(manifests / f"{split}.jsonl", "sha256")}
        atomic_json(manifests / "report.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
