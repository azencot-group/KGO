#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/../../datasets}"
exec python "$SCRIPT_DIR/../probts/utils/download_datasets.py" --data_path "$DATA_DIR"
