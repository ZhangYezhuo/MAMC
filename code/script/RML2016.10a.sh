#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Supply --data /path/to/dataset.pkl, or set AMC_DATA.
# For CPU use BACKEND=torch and pass --device cpu.
TRAIN_ARGS=(--backend "${BACKEND:-official}" --amp --epochs 100 --seed 42
            --output "$PROJECT_ROOT/outputs/rml2016-mamc2")
if [[ -n "${AMC_DATA:-}" ]]; then
  TRAIN_ARGS+=(--data "$AMC_DATA")
fi
exec python "$PROJECT_ROOT/scripts/train.py" "${TRAIN_ARGS[@]}" "$@"
