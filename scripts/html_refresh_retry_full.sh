#!/usr/bin/env bash
# Retry a subset of notebooks after full-defaults restore (failures / early kills).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs/html-refresh-full-retry"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TIMEOUT="${NOTEBOOK_TIMEOUT:-14400}"

mkdir -p "$LOG_DIR"

# Override via: PROJECTS_CSV="dir|nb,dir|nb" bash scripts/html_refresh_retry_full.sh
if [[ -n "${PROJECTS_CSV:-}" ]]; then
  IFS=',' read -r -a PROJECTS <<< "$PROJECTS_CSV"
else
  PROJECTS=(
    "AI_Cost_Cutting_Market_Analysis|AI_Cost_Cutting_Market_Analysis.ipynb"
    "Board_Management_Monitoring|Board_Management_Monitoring.ipynb"
  )
fi

run_one() {
  local entry="$1"
  local dir="${entry%%|*}"
  local notebook="${entry#*|}"
  local proj_dir="$ROOT/$dir"
  local log="$LOG_DIR/${dir}.log"

  {
    echo "=== RETRY START $(date -Iseconds) $dir ==="
    cd "$proj_dir"
    ln -sf "$ROOT/.env" .env
    if [[ ! -d .venv-html ]]; then
      uv venv .venv-html --python 3.13
      uv pip install --python .venv-html/bin/python -r requirements.txt nbconvert ipykernel tqdm nest-asyncio openpyxl
    fi
    set +u
    if [[ -f .env ]]; then set -a; # shellcheck disable=SC1091
      source .env; set +a; fi
    set -u
    .venv-html/bin/python "$ROOT/scripts/clear_notebook_outputs.py" "$notebook"
    .venv-html/bin/python -m jupyter nbconvert \
      --to notebook --execute --inplace "$notebook" \
      --ExecutePreprocessor.timeout="$TIMEOUT" \
      --ExecutePreprocessor.kernel_name=python3 \
      --NbConvertApp.validate_notebook=False
    .venv-html/bin/python "$ROOT/scripts/fix_notebook_schema.py" "$notebook"
    .venv-html/bin/python -m jupyter nbconvert --to html "$notebook"
    echo "=== RETRY DONE $(date -Iseconds) $dir ==="
  } >"$log" 2>&1
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then break; fi
    sleep 15
  done
}

for entry in "${PROJECTS[@]}"; do
  wait_for_slot
  run_one "$entry" &
  echo "Retry launched ${entry%%|*}"
done
wait
echo "Retry finished. Logs in $LOG_DIR"
