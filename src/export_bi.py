"""Score every customer and export clean tables for the Power BI dashboard and an Excel summary."""
import numpy as np
import pandas as pd

from config import PROCESSED_DIR, REPORTS_DIR
from features import FEATURES
from retention import RetentionAdvisor, label_for

BI_DIR = REPORTS_DIR / "powerbi"


def main():
    BI_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "customers_features.csv")
    seg = pd.read_csv(PROCESSED_DIR / "segments.csv")
    df = df.merge(seg, on="customer_id", how="left")
    for col in ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service"]:
        df[col] = df[col].astype(int)

    advisor = RetentionAdvisor()
    X = df[FEATURES]
    df["churn_probability"] = advisor.pipeline.predict_proba(X)[:, 1].round(4)
    df["risk_band"] = pd.cut(df["churn_probability"], [-0.01, 0.40, 0.70, 1.0],
                             labels=["low", "medium", "high"]).astype(str)

    # top SHAP driver per customer, labelled from the customer's actual feature values
    Xt = pd.DataFrame(advisor.prep.transform(X), columns=advisor.prep.get_feature_names_out())
    shap_vals = np.asarray(advisor.explainer.shap_values(Xt))
    cols = Xt.columns
    top_driver = []
    for i in range(len(Xt)):
        label = "none"
        for j in np.argsort(-shap_vals[i]):
            if shap_vals[i, j] <= 0:
                break
            label = label_for(cols[j], Xt.iat[i, j]) or label
            if label != "none":
                break
        top_driver.append(label)
    df["top_churn_driver"] = top_driver
    df["monthly_revenue_at_risk"] = (df["monthly_charges"] * df["churn_probability"]).round(2)

    scored_cols = ["customer_id", "gender", "senior_citizen", "tenure_months", "tenure_cohort",
                   "contract", "payment_method", "internet_service", "monthly_charges",
                   "addon_count", "segment", "churn", "churn_probability", "risk_band",
                   "top_churn_driver", "monthly_revenue_at_risk"]
    scored = df[scored_cols]
    scored.to_csv(BI_DIR / "customer_scores.csv", index=False)

    by_segment = (scored.groupby("segment")
                  .agg(customers=("customer_id", "count"), actual_churn_rate=("churn", "mean"),
                       avg_risk=("churn_probability", "mean"),
                       revenue_at_risk=("monthly_revenue_at_risk", "sum"))
                  .round(3).reset_index())
    by_contract = (scored.groupby(["contract", "risk_band"]).size()
                   .unstack(fill_value=0).reset_index())
    drivers = (scored[scored["risk_band"] == "high"]["top_churn_driver"]
               .value_counts().rename_axis("driver").reset_index(name="high_risk_customers"))

    by_segment.to_csv(BI_DIR / "segment_summary.csv", index=False)
    with pd.ExcelWriter(REPORTS_DIR / "retention_summary.xlsx") as xl:
        by_segment.to_excel(xl, sheet_name="By segment", index=False)
        by_contract.to_excel(xl, sheet_name="Contract x risk", index=False)
        drivers.to_excel(xl, sheet_name="High-risk drivers", index=False)
        scored.sort_values("churn_probability", ascending=False).head(500) \
              .to_excel(xl, sheet_name="Top 500 at risk", index=False)

    print(scored["risk_band"].value_counts().to_string())
    print("\nhigh-risk drivers:\n", drivers.to_string(index=False))
    print(f"\nmonthly revenue at risk: {scored['monthly_revenue_at_risk'].sum():,.0f}")


if __name__ == "__main__":
    main()
