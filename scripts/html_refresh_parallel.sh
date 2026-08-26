#!/usr/bin/env bash
# Execute migrated cookbooks and regenerate companion HTML exports (full defaults).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs/html-refresh"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TIMEOUT="${NOTEBOOK_TIMEOUT:-7200}"

mkdir -p "$LOG_DIR"

# dir|notebook filename
PROJECTS=(
  "AI_Cost_Cutting_Market_Analysis|AI_Cost_Cutting_Market_Analysis.ipynb"
  "AI_Revenue_Generation_Market_Analysis|AI_Revenue_Generation_Market_Analysis.ipynb"
  "Board_Management_Monitoring|Board_Management_Monitoring.ipynb"
  "Credit_Ratings_Monitoring|Credit_Ratings_Monitoring.ipynb"
  "Daily_Digest_Central_Banks|Daily_Digest_Central_Banks.ipynb"
  "Daily_Digest_Crude_Oil|Daily_Digest_Crude_Oil.ipynb"
  "Election_Monitor|Trump_Reelection_Impact_Analysis.ipynb"
  "Liquid_Cooling_Market_Watch|Liquid_Cooling_Market_Watch.ipynb"
  "Narrative_Miners|NarrativeMiner.ipynb"
  "Pricing_Power_Analysis|Pricing Power.ipynb"
  "Report_Generator_AI_Threats|Report Generator_ AI Disruption Risk.ipynb"
  "Report_Generator_Regulatory_Issues_in_Tech|Report Generator_ Regulatory Issues.ipynb"
  "Report_Generator_Specialized_Report_Tariffs|Report_Generator_Specialized_Report_Tariffs.ipynb"
  "Risk_Analyzer|Risk_Analyzer.ipynb"
  "Screener_for_Crypto|Screener_for_Crypto.ipynb"
  "Tracking_Inflation_Drivers|Tracking_Inflation_Drivers.ipynb"
  "Rising_Bond_Spread_Risks|Rising_Bond_Spread_Risks.ipynb"
)

run_one() {
  local entry="$1"
  local dir="${entry%%|*}"
  local notebook="${entry#*|}"
  local proj_dir="$ROOT/$dir"
  local log="$LOG_DIR/${dir}.log"

  {
    echo "=== START $(date -Iseconds) $dir ==="
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

    .venv-html/bin/python -m jupyter nbconvert --to html "$notebook"

    echo "=== DONE $(date -Iseconds) $dir ==="
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
  echo "Launched ${entry%%|*}"
done

wait
echo "All jobs finished. Logs in $LOG_DIR"
