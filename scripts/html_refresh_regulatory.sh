#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs/html-refresh-retry"
TIMEOUT="${NOTEBOOK_TIMEOUT:-7200}"
DIR="Report_Generator_Regulatory_Issues_in_Tech"
NOTEBOOK="Report Generator_ Regulatory Issues.ipynb"
proj_dir="$ROOT/$DIR"
log="$LOG_DIR/${DIR}-final.log"

{
  echo "=== REGULATORY FINAL START $(date -Iseconds) ==="
  cd "$proj_dir"
  ln -sf "$ROOT/.env" .env

  .venv-html/bin/python "$ROOT/scripts/clear_notebook_outputs.py" "$NOTEBOOK"

  set +u
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  set -u

  .venv-html/bin/python -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "$NOTEBOOK" \
    --ExecutePreprocessor.timeout="$TIMEOUT" \
    --ExecutePreprocessor.kernel_name=python3

  .venv-html/bin/python -m jupyter nbconvert --to html "$NOTEBOOK"

  echo "=== REGULATORY FINAL DONE $(date -Iseconds) ==="
} >"$log" 2>&1

echo "Finished. Log: $log"
