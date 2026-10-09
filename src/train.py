"""Train and compare churn models, explain the winner with SHAP, track runs in MLflow."""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MLFLOW_URI, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED
from features import FEATURES, TARGET, make_preprocessor


def load_xy():
    df = pd.read_csv(PROCESSED_DIR / "customers_features.csv")
    for col in ["senior_citizen", "partner", "dependents", "paperless_billing", "phone_service"]:
        df[col] = df[col].astype(int)
    return df, df[FEATURES], df[TARGET]


def best_f1_threshold(y_true, proba):
    """Default 0.5 is wrong for a 27% positive class; pick the threshold that maximises F1."""
    grid = np.arange(0.20, 0.71, 0.01)
    scores = [f1_score(y_true, proba >= t) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {"roc_auc": round(roc_auc_score(y_true, proba), 4),
            "pr_auc": round(average_precision_score(y_true, proba), 4),
            "precision": round(precision_score(y_true, pred), 4),
            "recall": round(recall_score(y_true, pred), 4),
            "f1": round(f1_score(y_true, pred), 4),
            "threshold": threshold}


def main():
    df, X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    imbalance = float((y_tr == 0).sum() / (y_tr == 1).sum())  # ~2.8 negatives per churner
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    candidates = {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "lightgbm": Pipeline([
            ("prep", make_preprocessor()),
            ("model", LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, scale_pos_weight=imbalance,
                                     random_state=SEED, verbose=-1))]),
    }

    # XGBoost gets a proper randomized search, it's the main model
    xgb_pipe = Pipeline([("prep", make_preprocessor()),
                         ("model", XGBClassifier(eval_metric="auc", scale_pos_weight=imbalance,
                                                 random_state=SEED, n_jobs=4))])
    search = RandomizedSearchCV(
        xgb_pipe,
        param_distributions={
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__reg_lambda": [0.5, 1, 2, 5],
        },
        n_iter=40, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1)
    search.fit(X_tr, y_tr)
    candidates["xgboost"] = search.best_estimator_

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn-intelligence")
    results = {}
    for name, pipe in candidates.items():
        with mlflow.start_run(run_name=name):
            cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
            pipe.fit(X_tr, y_tr)
            proba_tr = pipe.predict_proba(X_tr)[:, 1]
            threshold = best_f1_threshold(y_tr, proba_tr)
            test = evaluate(y_te, pipe.predict_proba(X_te)[:, 1], threshold)
            results[name] = {"cv_roc_auc_mean": round(cv_auc.mean(), 4),
                             "cv_roc_auc_std": round(cv_auc.std(), 4), "test": test}
            if name == "xgboost":
                mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_roc_auc", cv_auc.mean())
            mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
            print(f"{name:20s} cv_auc={cv_auc.mean():.4f}±{cv_auc.std():.4f}  test={test}")

    best_name = max(results, key=lambda k: results[k]["test"]["roc_auc"])
    best = candidates[best_name]
    joblib.dump({"pipeline": best, "threshold": results[best_name]["test"]["threshold"],
                 "model_name": best_name, "features": FEATURES}, MODELS_DIR / "churn_model.joblib")

    # SHAP on the XGBoost model: global drivers + a per-customer explanation file for the API
    xgb = candidates["xgboost"]
    X_te_t = pd.DataFrame(xgb.named_steps["prep"].transform(X_te),
                          columns=xgb.named_steps["prep"].get_feature_names_out())
    explainer = shap.TreeExplainer(xgb.named_steps["model"])
    shap_values = explainer.shap_values(X_te_t)
    importance = (pd.Series(np.abs(shap_values).mean(axis=0), index=X_te_t.columns)
                  .sort_values(ascending=False))
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    # the top-decile lift is what a retention team actually cares about
    proba_te = best.predict_proba(X_te)[:, 1]
    top_decile = y_te.to_numpy()[np.argsort(-proba_te)[: len(proba_te) // 10]]
    lift = round(top_decile.mean() / y_te.mean(), 2)

    report = {"models": results, "best_model": best_name,
              "best_params_xgboost": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
              "top_decile_churn_rate": round(float(top_decile.mean()), 4),
              "top_decile_lift": lift,
              "shap_top_features": {k: round(float(v), 4) for k, v in importance.head(12).items()}}
    (REPORTS_DIR / "model_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("best_model", "top_decile_churn_rate",
                                              "top_decile_lift", "shap_top_features")}, indent=2))


if __name__ == "__main__":
    main()
