#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'usage: scripts/status.sh /mnt/raid/asr-lab/runs/RUN_ID' >&2
  exit 2
fi
exec "${PYTHON:-python}" - "$1" <<'PY'
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
path = run / "monitor/status.json"
if not path.exists():
    print(json.dumps({"run": str(run), "status": "monitor_not_started"}))
    raise SystemExit(1)
status = json.loads(path.read_text())
status["monitor_age_seconds"] = round(time.time() - path.stat().st_mtime, 1)
if status["monitor_age_seconds"] > 90:
    status["alerts"] = list(status.get("alerts", [])) + ["monitor_stale"]
print(json.dumps(status, indent=2, allow_nan=False))
PY
