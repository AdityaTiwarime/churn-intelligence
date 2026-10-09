#!/usr/bin/env bash
# Full pipeline: database -> SQL analysis -> features -> stats -> models -> segments -> BI exports.
# Needs a running Postgres. Locally:  docker compose up -d db
#   export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/churn
set -euo pipefail
cd "$(dirname "$0")/src"
export MLFLOW_DISABLE_AGENT_HINT=1

python load_db.py
python sql_analysis.py
python prepare.py
python stats_tests.py
python train.py
python segment.py
python export_bi.py

echo "done. reports/ has every result, models/ has the trained model."
