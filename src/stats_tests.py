"""Hypothesis tests on churn drivers plus the design of a retention-offer A/B test.

The dataset has no real experiment in it, so the A/B section does two honest things:
1. sizes the experiment (power analysis) using the real baseline churn of the target segment
2. runs the planned analysis on a simulated assignment, to prove the pipeline works end to end
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from config import PROCESSED_DIR, REPORTS_DIR, SEED


def chi_square(df, col):
    table = pd.crosstab(df[col], df["churn"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": round(chi2, 1), "dof": int(dof), "p_value": float(f"{p:.3g}"),
            "cramers_v": round(cramers_v, 3)}


def charges_test(df):
    churned = df.loc[df["churn"] == 1, "monthly_charges"]
    stayed = df.loc[df["churn"] == 0, "monthly_charges"]
    t, p = stats.ttest_ind(churned, stayed, equal_var=False)  # Welch: variances differ
    u, p_mw = stats.mannwhitneyu(churned, stayed)
    return {"mean_churned": round(churned.mean(), 2), "mean_stayed": round(stayed.mean(), 2),
            "welch_t": round(t, 2), "p_value": float(f"{p:.3g}"),
            "mann_whitney_p": float(f"{p_mw:.3g}")}


def correlations(df):
    cols = ["tenure_months", "monthly_charges", "total_charges", "addon_count",
            "auto_pay", "is_fiber", "charge_increase", "churn"]
    corr = df[cols].corr()["churn"].drop("churn").sort_values()
    return {k: round(v, 3) for k, v in corr.items()}


def ab_test_design(df, relative_lift=0.20, alpha=0.05, power=0.80):
    """Retention offer for month-to-month customers: discount on a 1-year contract upgrade."""
    target = df[df["contract"] == "Month-to-month"]
    p_control = target["churn"].mean()
    p_treat = p_control * (1 - relative_lift)
    effect = proportion_effectsize(p_control, p_treat)
    n_per_arm = int(np.ceil(NormalIndPower().solve_power(effect, alpha=alpha, power=power, ratio=1)))

    # simulated run: randomly split the real segment, apply the assumed lift to the treatment arm
    rng = np.random.default_rng(SEED)
    n = min(n_per_arm, len(target) // 2)
    sample = target.sample(2 * n, random_state=SEED)
    control, treatment = sample.iloc[:n], sample.iloc[n:]
    treat_churn = treatment["churn"].to_numpy().copy()
    saved = rng.random(len(treat_churn)) < relative_lift
    treat_churn[(treat_churn == 1) & saved] = 0

    counts = np.array([control["churn"].sum(), treat_churn.sum()])
    nobs = np.array([n, n])
    z, p = proportions_ztest(counts, nobs)
    return {
        "segment": "Month-to-month customers",
        "segment_size": int(len(target)),
        "baseline_churn": round(p_control, 4),
        "target_churn_with_offer": round(p_treat, 4),
        "alpha": alpha, "power": power,
        "required_n_per_arm": n_per_arm,
        "simulated": {"n_per_arm": int(n),
                      "control_churn": round(counts[0] / n, 4),
                      "treatment_churn": round(counts[1] / n, 4),
                      "z": round(z, 2), "p_value": float(f"{p:.3g}"),
                      "note": "treatment lift is assumed, not observed; this validates the analysis plan"},
    }


def main():
    df = pd.read_csv(PROCESSED_DIR / "customers_features.csv")
    results = {
        "chi_square": {c: chi_square(df, c) for c in
                       ["contract", "payment_method", "internet_service", "tech_support",
                        "senior_citizen", "gender"]},
        "monthly_charges_test": charges_test(df),
        "correlation_with_churn": correlations(df),
        "descriptive": df[["tenure_months", "monthly_charges", "total_charges"]]
                         .describe().round(2).to_dict(),
        "ab_test": ab_test_design(df),
    }
    (REPORTS_DIR / "stats_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: results[k] for k in ("chi_square", "monthly_charges_test",
                                              "correlation_with_churn", "ab_test")}, indent=2))


if __name__ == "__main__":
    main()
