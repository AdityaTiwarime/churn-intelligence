import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from main import app  # noqa: E402

client = TestClient(app)

RISKY = {
    "tenure_months": 2, "contract": "Month-to-month", "payment_method": "Electronic check",
    "monthly_charges": 95.5, "total_charges": 190.0, "internet_service": "Fiber optic",
    "senior_citizen": True,
}
SAFE = {
    "tenure_months": 64, "contract": "Two year", "payment_method": "Credit card (automatic)",
    "monthly_charges": 24.0, "total_charges": 1530.0, "internet_service": "No",
    "partner": True, "dependents": True,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_risky_customer_scores_high_with_reasons():
    body = client.post("/predict", json=RISKY).json()
    assert body["churn_probability"] > 0.6
    assert body["risk_band"] in {"medium", "high"}
    assert body["top_drivers"], "expected SHAP reasons for a risky customer"
    assert body["retention_plan"]["actions"]


def test_safe_customer_scores_low():
    body = client.post("/predict", json=SAFE).json()
    assert body["churn_probability"] < 0.2
    assert body["risk_band"] == "low"


def test_bad_input_rejected():
    r = client.post("/predict", json={**RISKY, "contract": "Weekly"})
    assert r.status_code == 422


def test_reasons_match_the_customer():
    # internet-only customer: a phone-related reason must describe "no phone line", never the opposite
    body = client.post("/predict", json={**RISKY, "phone_service": False}).json()
    reasons = [d["reason"] for d in body["top_drivers"]]
    assert "has a phone line" not in reasons
