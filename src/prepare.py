"""Pull the joined customer view from Postgres, clean it and engineer features."""
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL, PROCESSED_DIR, REPORTS_DIR

JOINED_VIEW = """
SELECT c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
       a.tenure_months, a.contract, a.paperless_billing, a.payment_method,
       a.monthly_charges, a.total_charges, a.churned,
       s.phone_service, s.multiple_lines, s.internet_service, s.online_security,
       s.online_backup, s.device_protection, s.tech_support, s.streaming_tv,
       s.streaming_movies
FROM customers c
JOIN accounts a ON a.customer_id = c.customer_id
JOIN services s ON s.customer_id = c.customer_id
ORDER BY c.customer_id
"""

ADDON_COLS = ["online_security", "online_backup", "device_protection",
              "tech_support", "streaming_tv", "streaming_movies"]


def load_joined() -> pd.DataFrame:
    df = pd.read_sql(JOINED_VIEW, create_engine(DATABASE_URL))
    for col in ("monthly_charges", "total_charges"):
        df[col] = df[col].astype(float)
    return df


def outlier_report(df: pd.DataFrame, cols) -> dict:
    """IQR check. We report outliers instead of silently dropping customers."""
    report = {}
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report[col] = {"lower": round(lo, 2), "upper": round(hi, 2),
                       "outliers": int(((df[col] < lo) | (df[col] > hi)).sum())}
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customer_id").copy()
    # "No internet service" / "No phone service" just mean the add-on isn't there
    for col in ADDON_COLS + ["multiple_lines"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["addon_count"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["has_family"] = (df["partner"] | df["dependents"]).astype(int)
    df["auto_pay"] = df["payment_method"].str.contains("automatic").astype(int)
    df["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
    df["new_customer"] = (df["tenure_months"] <= 6).astype(int)
    # what they actually paid per month vs today's bill -> catches recent price jumps
    df["avg_monthly_paid"] = np.where(df["tenure_months"] > 0,
                                      df["total_charges"] / df["tenure_months"].clip(lower=1),
                                      df["monthly_charges"])
    df["charge_increase"] = df["monthly_charges"] - df["avg_monthly_paid"]
    df["tenure_cohort"] = pd.cut(df["tenure_months"], [-1, 6, 12, 24, 48, 72],
                                 labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype(str)
    df["churn"] = df["churned"].astype(int)
    return df


def main():
    raw = load_joined()
    df = engineer(clean(raw))
    df.to_csv(PROCESSED_DIR / "customers_features.csv", index=False)

    summary = {
        "rows": len(df),
        "churn_rate": round(df["churn"].mean(), 4),
        "outliers_iqr": outlier_report(df, ["tenure_months", "monthly_charges", "total_charges"]),
        "engineered_features": ["addon_count", "has_family", "auto_pay", "is_fiber",
                                "new_customer", "avg_monthly_paid", "charge_increase", "tenure_cohort"],
    }
    (REPORTS_DIR / "data_prep.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
