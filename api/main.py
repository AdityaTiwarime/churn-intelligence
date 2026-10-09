"""FastAPI service: churn probability, risk band, SHAP reasons and a retention plan."""
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURES  # noqa: E402
from prepare import clean, engineer  # noqa: E402
from retention import RetentionAdvisor  # noqa: E402

YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]

app = FastAPI(title="Churn Intelligence API", version="1.0.0")
advisor = RetentionAdvisor()


class Customer(BaseModel):
    customer_id: str = "API-CUSTOMER"
    gender: Literal["Male", "Female"] = "Female"
    senior_citizen: bool = False
    partner: bool = False
    dependents: bool = False
    tenure_months: int = Field(ge=0, le=100)
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: bool = True
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(gt=0)
    total_charges: float = Field(ge=0)
    phone_service: bool = True
    multiple_lines: Literal["Yes", "No", "No phone service"] = "No"
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Addon = "No"
    online_backup: Addon = "No"
    device_protection: Addon = "No"
    tech_support: Addon = "No"
    streaming_tv: Addon = "No"
    streaming_movies: Addon = "No"


def to_features(c: Customer) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump()])
    df["churned"] = False  # engineer() expects the column; it isn't used for prediction
    df = engineer(clean(df))
    for col in ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service"]:
        df[col] = df[col].astype(int)
    return df[FEATURES]


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "decision_threshold": advisor.threshold}


@app.post("/predict")
def predict(customer: Customer):
    X = to_features(customer)
    proba = float(advisor.pipeline.predict_proba(X)[:, 1][0])
    drivers = advisor.drivers(X)
    return {
        "customer_id": customer.customer_id,
        "churn_probability": round(proba, 4),
        "will_churn": proba >= advisor.threshold,
        "risk_band": risk_band(proba),
        "top_drivers": drivers,
        "retention_plan": advisor.plan(proba, drivers),
    }
