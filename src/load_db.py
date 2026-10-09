"""Load the raw telco CSV into the normalized Postgres schema."""
import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, RAW_CSV, SQL_DIR

YES_NO = {"Yes": True, "No": False}


def read_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    # 11 brand-new customers have a blank TotalCharges; they simply haven't been billed yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    return df


def split_tables(df: pd.DataFrame):
    customers = pd.DataFrame({
        "customer_id": df["customerID"],
        "gender": df["gender"],
        "senior_citizen": df["SeniorCitizen"].astype(bool),
        "partner": df["Partner"].map(YES_NO),
        "dependents": df["Dependents"].map(YES_NO),
    })
    accounts = pd.DataFrame({
        "customer_id": df["customerID"],
        "tenure_months": df["tenure"],
        "contract": df["Contract"],
        "paperless_billing": df["PaperlessBilling"].map(YES_NO),
        "payment_method": df["PaymentMethod"],
        "monthly_charges": df["MonthlyCharges"],
        "total_charges": df["TotalCharges"],
        "churned": df["Churn"].map(YES_NO),
    })
    services = pd.DataFrame({
        "customer_id": df["customerID"],
        "phone_service": df["PhoneService"].map(YES_NO),
        "multiple_lines": df["MultipleLines"],
        "internet_service": df["InternetService"],
        "online_security": df["OnlineSecurity"],
        "online_backup": df["OnlineBackup"],
        "device_protection": df["DeviceProtection"],
        "tech_support": df["TechSupport"],
        "streaming_tv": df["StreamingTV"],
        "streaming_movies": df["StreamingMovies"],
    })
    return customers, accounts, services


def main():
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text((SQL_DIR / "01_schema.sql").read_text()))

    customers, accounts, services = split_tables(read_raw())
    # parent table first so the foreign keys are satisfied
    for name, frame in (("customers", customers), ("accounts", accounts), ("services", services)):
        frame.to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
        print(f"loaded {len(frame):>5} rows into {name}")


if __name__ == "__main__":
    main()
