"""Turn a customer's SHAP churn drivers into a retention plan.

With OPENAI_API_KEY set, the plan is written by an LLM from the drivers.
Without it (local runs, CI, tests) a rule-based plan is returned, so nothing breaks.
"""
import os

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR

# plain-English names for the encoded features the model actually sees
FRIENDLY = {
    "contract_Month-to-month": "on a month-to-month contract",
    "contract_Two year": "no long-term contract",
    "contract_One year": "no long-term contract",
    "tenure_months": "short time with the company",
    "internet_service_Fiber optic": "fiber internet plan",
    "internet_service_No": "no internet service",
    "payment_method_Electronic check": "pays by electronic check",
    "monthly_charges": "high monthly bill",
    "total_charges": "low lifetime spend",
    "online_security_No": "no online security add-on",
    "tech_support_No": "no tech support add-on",
    "online_backup_No": "no online backup add-on",
    "streaming_movies_No": "no streaming add-ons",
    "streaming_tv_No": "no streaming add-ons",
}

# yes/no features need a label for each value: SHAP can push risk up for either one
BINARY_LABELS = {
    "senior_citizen": {1: "senior citizen", 0: "not a senior citizen"},
    "partner": {1: "has a partner", 0: "no partner"},
    "dependents": {1: "has dependents", 0: "no dependents"},
    "paperless_billing": {1: "paperless billing", 0: "paper billing"},
    "phone_service": {1: "has a phone line", 0: "internet-only plan"},
    "has_family": {1: "family on the account", 0: "no partner or dependents"},
    "auto_pay": {1: "on automatic payment", 0: "not on automatic payment"},
    "is_fiber": {1: "fiber internet plan", 0: "not on fiber"},
    "new_customer": {1: "joined in the last 6 months", 0: "long-standing customer"},
}


def label_for(feature: str, value: float):
    """Plain-English reason for one feature, or None if it shouldn't be shown."""
    if feature in BINARY_LABELS:
        return BINARY_LABELS[feature][int(round(value))]
    if feature in FRIENDLY and "_" in feature and feature.split("_")[0] in ONE_HOT_PREFIXES:
        # one-hot column: only meaningful when the customer actually has that category
        return FRIENDLY[feature] if value == 1 else None
    if feature in FRIENDLY:
        return FRIENDLY[feature]
    return feature.replace("_", " ") if value == 1 else None


ONE_HOT_PREFIXES = {"contract", "payment", "internet", "online", "tech", "streaming",
                    "device", "multiple", "tenure"}

RULES = {
    "on a month-to-month contract": "Offer a discounted upgrade to a 1-year contract.",
    "no long-term contract": "Offer a discounted upgrade to a 1-year contract.",
    "short time with the company": "Enrol in a 90-day onboarding check-in programme.",
    "joined in the last 6 months": "Enrol in a 90-day onboarding check-in programme.",
    "fiber internet plan": "Run a fiber service-quality check and offer a speed or price review.",
    "pays by electronic check": "Offer a small credit for switching to automatic payment.",
    "not on automatic payment": "Offer a small credit for switching to automatic payment.",
    "paper billing": "Offer e-billing with a small one-time credit.",
    "high monthly bill": "Review the plan for a better-value bundle.",
    "no tech support add-on": "Offer 3 months of free tech support.",
    "no online security add-on": "Offer a free online security trial.",
}


class RetentionAdvisor:
    def __init__(self):
        bundle = joblib.load(MODELS_DIR / "churn_model.joblib")
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        self.explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
        # the explainer was built on the XGBoost model; reuse its preprocessor for feature names
        self.prep = self.pipeline.named_steps["prep"]

    def drivers(self, X: pd.DataFrame, top_n: int = 4):
        Xt = pd.DataFrame(self.prep.transform(X), columns=self.prep.get_feature_names_out())
        contrib = pd.Series(np.asarray(self.explainer.shap_values(Xt))[0], index=Xt.columns)
        pushing_up = contrib[contrib > 0].sort_values(ascending=False)
        row = Xt.iloc[0]
        seen, out = set(), []
        for feat, val in pushing_up.items():
            label = label_for(feat, row[feat])
            if label and label not in seen:
                seen.add(label)
                out.append({"feature": feat, "reason": label, "impact": round(float(val), 3)})
            if len(out) == top_n:
                break
        return out

    def plan(self, probability: float, drivers: list) -> dict:
        if os.getenv("OPENAI_API_KEY"):
            try:
                return {"source": "llm", "actions": self._llm_plan(probability, drivers)}
            except Exception as exc:  # network or quota problems shouldn't kill the prediction
                fallback = self._rule_plan(drivers)
                return {"source": "rules", "actions": fallback, "llm_error": type(exc).__name__}
        return {"source": "rules", "actions": self._rule_plan(drivers)}

    @staticmethod
    def _rule_plan(drivers):
        actions = []
        for d in drivers:
            action = RULES.get(d["reason"])
            if action and action not in actions:
                actions.append(action)
        return actions or ["Schedule a personal check-in call from the retention team."]

    @staticmethod
    def _llm_plan(probability, drivers):
        from openai import OpenAI

        reasons = "; ".join(d["reason"] for d in drivers)
        prompt = (
            "You are a retention specialist at a telecom company.\n"
            f"A customer has a {probability:.0%} predicted chance of leaving.\n"
            f"The model's main reasons: {reasons}.\n"
            "Write exactly 3 short, specific retention actions, one per line, no numbering. "
            "Only use offers a telecom can realistically make (contract discounts, add-on trials, "
            "payment credits, plan reviews, support calls). Do not invent customer details."
        )
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=200)
        lines = [ln.strip("-• ").strip() for ln in resp.choices[0].message.content.splitlines()]
        return [ln for ln in lines if ln][:3]
