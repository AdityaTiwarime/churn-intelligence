"""K-means customer segments, picked by silhouette score and visualised with PCA."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DIR, REPORTS_DIR, SEED

# behaviour only - churn itself is NOT a clustering input, it's used afterwards to profile segments
SEGMENT_COLS = ["tenure_months", "monthly_charges", "addon_count", "auto_pay", "is_fiber",
                "has_family", "senior_citizen"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "customers_features.csv")
    X = df[SEGMENT_COLS].astype(float).copy()
    X["long_contract"] = (df["contract"] != "Month-to-month").astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    scores = {}
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_scaled)
        scores[k] = round(silhouette_score(X_scaled, labels, sample_size=3000, random_state=SEED), 4)
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X_scaled)
    df["segment"] = km.labels_

    profile = (df.groupby("segment")
                 .agg(customers=("customer_id", "count"),
                      churn_rate=("churn", "mean"),
                      avg_tenure=("tenure_months", "mean"),
                      avg_monthly=("monthly_charges", "mean"),
                      fiber_share=("is_fiber", "mean"),
                      auto_pay_share=("auto_pay", "mean"),
                      avg_addons=("addon_count", "mean"))
                 .round(3).sort_values("churn_rate", ascending=False))

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg in sorted(df["segment"].unique()):
        m = df["segment"] == seg
        rate = profile.loc[seg, "churn_rate"]
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5, label=f"Segment {seg} (churn {rate:.0%})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.set_title("Customer segments (k-means, PCA view)")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "segments_pca.png", dpi=150)

    df[["customer_id", "segment"]].to_csv(PROCESSED_DIR / "segments.csv", index=False)
    out = {"silhouette_by_k": scores, "best_k": best_k,
           "pca_variance_explained": [round(v, 3) for v in pca.explained_variance_ratio_],
           "profile": json.loads(profile.reset_index().to_json(orient="records"))}
    (REPORTS_DIR / "segments.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("silhouette_by_k", "best_k", "pca_variance_explained")}))
    print(profile.to_string())


if __name__ == "__main__":
    main()
