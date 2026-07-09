#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
from src.data.dataset import load_and_prepare_data

load_and_prepare_data()
PY