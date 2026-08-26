#!/usr/bin/env bash
# Re-run notebooks that failed due to legacy invalid output metadata.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs/html-refresh-retry"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TIMEOUT="${NOTEBOOK_TIMEOUT:-7200}"

mkdir -p "$LOG_DIR"

PROJECTS=(
  "Liquid_Cooling_Market_Watch|Liquid_Cooling_Market_Watch.ipynb"
  "Narrative_Miners|NarrativeMiner.ipynb"
  "Pricing_Power_Analysis|Pricing Power.ipynb"
)

run_one() {
  local entry="$1"
  local dir="${entry%%|*}"
  local notebook="${entry#*|}"
  local proj_dir="$ROOT/$dir"
  local log="$LOG_DIR/${dir}-cleared.log"

  {
    echo "=== CLEARED RETRY START $(date -Iseconds) $dir ==="
    cd "$proj_dir"
    ln -sf "$ROOT/.env" .env

    if [[ ! -d .venv-html ]]; then
      uv venv .venv-html --python 3.13
      uv pip install --python .venv-html/bin/python -r requirements.txt nbconvert ipykernel tqdm nest-asyncio openpyxl
    fi

    set +u
    if [[ -f .env ]]; then
      set -a
      # shellcheck disable=SC1091
      source .env
      set +a
    fi
    set -u

    .venv-html/bin/python "$ROOT/scripts/clear_notebook_outputs.py" "$notebook"

    .venv-html/bin/python -m jupyter nbconvert \
      --to notebook \
      --execute \
      --inplace \
      "$notebook" \
      --ExecutePreprocessor.timeout="$TIMEOUT" \
      --ExecutePreprocessor.kernel_name=python3 \
      --NbConvertApp.validate_notebook=False

    .venv-html/bin/python -m jupyter nbconvert --to html "$notebook"

    echo "=== CLEARED RETRY DONE $(date -Iseconds) $dir ==="
  } >"$log" 2>&1
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 15
  done
}

for entry in "${PROJECTS[@]}"; do
  wait_for_slot
  run_one "$entry" &
  echo "Cleared retry launched ${entry%%|*}"
done

wait
echo "Cleared retry finished. Logs in $LOG_DIR"
