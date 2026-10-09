import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"

# Local default points at a unix-socket Postgres; docker-compose overrides it.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres@/churn?host=/tmp")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

SEED = 42

for d in (PROCESSED_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
