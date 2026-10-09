"""Run every named query in sql/02_analysis.sql and save the results."""
import json
import re

import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL, REPORTS_DIR, SQL_DIR


def named_queries(path):
    blocks = re.split(r"^-- name: (\w+)\s*$", path.read_text(), flags=re.M)
    # re.split gives ['', name1, body1, name2, body2, ...]
    return {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}


def main():
    engine = create_engine(DATABASE_URL)
    results = {}
    out_dir = REPORTS_DIR / "sql"
    out_dir.mkdir(exist_ok=True)

    for name, query in named_queries(SQL_DIR / "02_analysis.sql").items():
        df = pd.read_sql(query, engine)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        results[name] = json.loads(df.to_json(orient="records"))
        print(f"\n== {name}\n{df.to_string(index=False)}")

    (REPORTS_DIR / "sql_findings.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
