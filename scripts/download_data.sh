#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/mnt/raid/asr-lab/tmp}"
mkdir -p "$TMPDIR"
exec "${PYTHON:-python}" -m asr_lab.prepare_data all "$@"
