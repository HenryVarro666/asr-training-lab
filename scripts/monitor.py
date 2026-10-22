#!/usr/bin/env python3
"""Observe one ASR run and host resources without controlling other processes."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

GIB = 1024**3


def process_state(pid):
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split(") ", 1)[1].startswith("Z"):
            return "zombie"
        return "running"
    except (ProcessLookupError, FileNotFoundError):
        return "exited"
    except PermissionError:
        return "running_permission_denied"


def gpu_resources():
    fields = "index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        resources = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            keys = ("index", "uuid", "utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c", "power_w")
            item = dict(zip(keys, row))
            for key in keys:
                if key == "uuid":
                    continue
                try:
                    item[key] = float(item[key])
                except (ValueError, TypeError):
                    item[key] = None
            resources.append(item)
        return resources, None
    except (OSError, subprocess.SubprocessError) as error:
        return [], str(error)


def latest_metric(path):
    if not path.exists():
        return None, None
    # Read the last64KiB; tolerate the writer's unfinished final JSON line.
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - 65536))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                return record, time.time() - path.stat().st_mtime
        except (ValueError, OSError):
            continue
    return None, time.time() - path.stat().st_mtime


def numeric_loss_values(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if "loss" in key.lower() and isinstance(item, (int, float)):
                found.append((name, item))
            elif isinstance(item, dict):
                found.extend(numeric_loss_values(item, name))
    return found


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def snapshot(args):
    now = time.time()
    disk = shutil.disk_usage(args.project_root)
    gpus, gpu_error = gpu_resources()
    metric, metric_age = latest_metric(args.metrics or args.run_dir / "metrics.jsonl")
    checkpoints = [p for pattern in ("checkpoint.json", "*.pt", "*.pth", "*.ckpt")
                   for p in args.run_dir.rglob(pattern)
                   if p.is_file() and not any(part.endswith(".tmp") for part in p.parts)]
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None
    alerts = []
    state = process_state(args.pid)
    training_status = read_json(args.run_dir / "status.json")
    summary = read_json(args.run_dir / "summary.json")
    run_metadata = read_json(args.run_dir / "run.json") or read_json(args.run_dir / "run_metadata.json")
    terminal_status = (training_status or {}).get("status") or (summary or {}).get("status")
    if state in ("exited", "zombie") and terminal_status not in ("complete", "completed", "success"):
        alerts.append("training_process_exited")
    if terminal_status in ("failed", "error", "interrupted"):
        alerts.append("training_" + terminal_status)
    if metric_age is not None and metric_age > args.stale_seconds and state.startswith("running"):
        alerts.append("metrics_stale")
    if metric is None and state.startswith("running") and now - args.started_at > args.stale_seconds:
        alerts.append("no_metrics_yet")
    if metric and any(not math.isfinite(v) for _, v in numeric_loss_values(metric)):
        alerts.append("nonfinite_loss")
    if disk.free < args.reserve_gib * GIB:
        alerts.append("disk_reserve_low")
    if gpu_error:
        alerts.append("gpu_query_failed")
    checkpoint_age = None if latest is None else now - latest.stat().st_mtime
    if args.checkpoint_stale_seconds > 0 and state.startswith("running"):
        if checkpoint_age is not None and checkpoint_age > args.checkpoint_stale_seconds:
            alerts.append("checkpoint_stale")
        elif latest is None and now - args.started_at > args.checkpoint_stale_seconds:
            alerts.append("no_checkpoint_yet")
    return {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "unix_time": now,
            "training_pid": args.pid, "training_process": state, "gpu_target": args.gpu,
            "training_status": training_status, "summary": summary, "run_metadata": run_metadata,
            "gpus": gpus, "gpu_error": gpu_error, "disk_free_gib": disk.free / GIB,
            "disk_used_gib": disk.used / GIB, "latest_metric": metric,
            "metric_age_seconds": metric_age, "latest_checkpoint": str(latest) if latest else None,
            "checkpoint_age_seconds": checkpoint_age, "alerts": alerts}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/raid/asr-lab"))
    parser.add_argument("--pid", type=int)
    parser.add_argument("--gpu", default="0,1", help="Physical training GPU index; all GPUs observed")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--stale-seconds", type=float, default=900)
    parser.add_argument("--checkpoint-stale-seconds", type=float, default=7200)
    parser.add_argument("--reserve-gib", type=float, default=500)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    args.started_at = time.time()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "monitor"
    output.mkdir(exist_ok=True)
    csv_path = output / "resources.csv"
    fields = ["timestamp", "training_pid", "training_process", "gpu_index", "gpu_uuid", "gpu_utilization_percent",
              "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_temperature_c", "gpu_power_w",
              "disk_free_gib", "metric_age_seconds", "checkpoint_age_seconds", "alerts"]
    try:
        while True:
            status = snapshot(args)
            with (output / "samples.jsonl").open("a") as handle:
                # JSON null marks any nonfinite values; explicit alert retains the failure information.
                handle.write(json.dumps(json_safe(status), default=str, allow_nan=False) + "\n")
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(json_safe(status), indent=2, default=str, allow_nan=False) + "\n")
            os.replace(temporary, output / "status.json")
            new = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if new:
                    writer.writeheader()
                for gpu in status["gpus"] or [{}]:
                    writer.writerow({"timestamp": status["timestamp"], "training_pid": args.pid,
                        "training_process": status["training_process"], "gpu_index": gpu.get("index"),
                        "gpu_uuid": gpu.get("uuid"), "gpu_utilization_percent": gpu.get("utilization_percent"),
                        "gpu_memory_used_mib": gpu.get("memory_used_mib"), "gpu_memory_total_mib": gpu.get("memory_total_mib"),
                        "gpu_temperature_c": gpu.get("temperature_c"), "gpu_power_w": gpu.get("power_w"),
                        "disk_free_gib": status["disk_free_gib"], "metric_age_seconds": status["metric_age_seconds"],
                        "checkpoint_age_seconds": status["checkpoint_age_seconds"], "alerts": ";".join(status["alerts"])})
            print(json.dumps({k: status[k] for k in ("timestamp", "training_process", "disk_free_gib", "metric_age_seconds", "alerts")}), flush=True)
            if args.once or status["training_process"] in ("exited", "zombie"):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
