#!/usr/bin/env bash
# Retry failed notebook HTML refresh jobs (subset).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs/html-refresh-retry"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TIMEOUT="${NOTEBOOK_TIMEOUT:-7200}"

mkdir -p "$LOG_DIR"

PROJECTS=(
  "AI_Cost_Cutting_Market_Analysis|AI_Cost_Cutting_Market_Analysis.ipynb"
  "AI_Revenue_Generation_Market_Analysis|AI_Revenue_Generation_Market_Analysis.ipynb"
  "Daily_Digest_Crude_Oil|Daily_Digest_Crude_Oil.ipynb"
  "Liquid_Cooling_Market_Watch|Liquid_Cooling_Market_Watch.ipynb"
  "Narrative_Miners|NarrativeMiner.ipynb"
  "Pricing_Power_Analysis|Pricing Power.ipynb"
  "Report_Generator_Regulatory_Issues_in_Tech|Report Generator_ Regulatory Issues.ipynb"
  "Report_Generator_Specialized_Report_Tariffs|Report_Generator_Specialized_Report_Tariffs.ipynb"
  "Risk_Analyzer|Risk_Analyzer.ipynb"
  "Rising_Bond_Spread_Risks|Rising_Bond_Spread_Risks.ipynb"
)

run_one() {
  local entry="$1"
  local dir="${entry%%|*}"
  local notebook="${entry#*|}"
  local proj_dir="$ROOT/$dir"
  local log="$LOG_DIR/${dir}.log"

  {
    echo "=== RETRY START $(date -Iseconds) $dir ==="
    cd "$proj_dir"

    if [[ ! -f "$notebook" ]]; then
      echo "ERROR: missing notebook $notebook"
      return 1
    fi

    ln -sf "$ROOT/.env" .env

    if [[ ! -d .venv-html ]]; then
      uv venv .venv-html --python 3.13
    fi

    uv pip install --python .venv-html/bin/python -r requirements.txt nbconvert ipykernel tqdm nest-asyncio openpyxl

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

    .venv-html/bin/python "$ROOT/scripts/fix_notebook_schema.py" "$notebook"

    .venv-html/bin/python -m jupyter nbconvert --to html "$notebook"

    echo "=== RETRY DONE $(date -Iseconds) $dir ==="
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
  echo "Retry launched ${entry%%|*}"
done

wait
echo "Retry jobs finished. Logs in $LOG_DIR"
