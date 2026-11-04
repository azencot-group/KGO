#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  echo "Usage: bash scripts/runner.sh DATASET HORIZON SEED [runner options]"
  echo "Example: bash scripts/runner.sh etth1 720 0"
  echo "Grid: bash scripts/runner.sh path/to/grid.yaml [runner options]"
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (($# == 0)); then
  exec python "$SCRIPT_DIR/runner.py" "$SCRIPT_DIR/LTSF.yaml"
fi
case "$1" in
  *.yaml|*.yml)
    CONFIG_PATH="$1"
    shift
    exec python "$SCRIPT_DIR/runner.py" "$CONFIG_PATH" "$@"
    ;;
esac
if (($# < 3)); then
  usage >&2
  exit 2
fi
if [[ ! "$2" =~ ^[1-9][0-9]*$ || ! "$3" =~ ^[0-9]+$ ]]; then
  echo "HORIZON must be a positive integer and SEED a nonnegative integer." >&2
  exit 2
fi
DATASET="$1"
HORIZON="$2"
SEED="$3"
shift 3
# Short-term settings live in stsf.yaml (e.g. "etth1 24", "solar 24").
GRID="$SCRIPT_DIR/LTSF.yaml"
python "$SCRIPT_DIR/runner.py" "$GRID" --dataset "$DATASET" --horizon "$HORIZON" \
  --seed "$SEED" --dry-run >/dev/null 2>&1 || GRID="$SCRIPT_DIR/stsf.yaml"
exec python "$SCRIPT_DIR/runner.py" "$GRID" \
  --dataset "$DATASET" --horizon "$HORIZON" --seed "$SEED" "$@"
