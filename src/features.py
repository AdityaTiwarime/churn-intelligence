"""Single source of truth for model inputs, shared by training and the API."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["contract", "payment_method", "internet_service", "multiple_lines",
               "online_security", "online_backup", "device_protection", "tech_support",
               "streaming_tv", "streaming_movies", "tenure_cohort"]
NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "addon_count",
           "charge_increase", "avg_monthly_paid"]
BINARY = ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service",
          "has_family", "auto_pay", "is_fiber", "new_customer"]

FEATURES = CATEGORICAL + NUMERIC + BINARY
TARGET = "churn"


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("bin", "passthrough", BINARY),
    ], verbose_feature_names_out=False)
