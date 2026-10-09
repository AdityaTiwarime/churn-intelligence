# Customer Churn Intelligence Platform

Which telecom customers are about to leave, why, and what should the company do about it?

**Live API:** https://churn-intelligence-34rc.onrender.com (interactive docs; the free server may take ~50 seconds to wake up)

This project answers that end to end on the public **IBM Telco Customer Churn** dataset (7,043 customers): SQL analysis in PostgreSQL, statistics, a tuned churn model with per-customer explanations, customer segments, an LLM-written retention plan, a REST API and Power BI / Excel outputs for business users.

```
CSV ─► PostgreSQL (3 normalized tables) ─► SQL analysis
                     │
                     ▼
        cleaning + feature engineering ─► hypothesis tests + A/B test design
                     │
                     ▼
   Logistic Regression vs LightGBM vs tuned XGBoost  (MLflow tracked)
                     │
          SHAP drivers ─► retention plan (OpenAI, rule fallback)
                     │
       FastAPI /predict  ·  k-means segments  ·  Power BI + Excel exports
```

## Key results

| | |
|---|---|
| Overall churn | **26.5%** (1,869 of 7,043) |
| Best model | Tuned **XGBoost**, test **ROC-AUC 0.849** (5-fold CV 0.849 ± 0.015), PR-AUC 0.676 |
| Baselines | Logistic regression 0.846, LightGBM 0.839 |
| Top-decile lift | The 10% of customers ranked riskiest churn at **77%**, **2.9×** the base rate |
| At the chosen threshold | Recall 0.71, precision 0.57, F1 0.63 (threshold tuned for F1, not left at 0.5) |
| Revenue concentration | Month-to-month **fiber** customers are **72%** of all lost monthly revenue |
| Segments | 7 k-means segments; churn ranges from **2%** to **57%** |

Honest note: tuned XGBoost only edges out logistic regression on this dataset. The value of the tree model here is the SHAP explanations and the better PR-AUC, not a big jump in ROC-AUC.

## What the data says

From the SQL layer (`sql/02_analysis.sql`, results in `reports/sql/`):

- Month-to-month contracts churn at **42.7%** vs **2.8%** on two-year contracts.
- New customers (0–6 months) churn at **52.9%**; customers past 4 years at **9.5%**.
- Fiber customers **without tech support** churn at **49.4%**; with tech support, **22.6%**.
- Electronic-check payers churn at **45.3%**, automatic payers at ~16%.
- Senior citizens churn at **41.7%** vs 23.6%.

Statistics (`reports/stats_results.json`):

- Contract type is the strongest single driver (chi-square, Cramér's V = 0.41, p < 1e-250).
- **Gender has no effect** on churn (p = 0.49), so it is not something to target.
- Churned customers pay more per month: $74.4 vs $61.3 (Welch t = 18.4, p < 1e-70).

SHAP (global drivers of the XGBoost model): month-to-month contract, short tenure, fiber internet, no long-term contract, electronic-check payment.

## Retention offer A/B test (designed, not run)

The dataset contains no real experiment, so this part is a **design**, clearly labelled as such:

- Target: month-to-month customers (3,875; baseline churn 42.7%).
- Offer: discounted upgrade to a 1-year contract. Goal: a 20% relative drop in churn (42.7% → 34.2%).
- Power analysis (α = 0.05, power = 0.80): **508 customers per arm**.
- A simulated run on random assignment checks the analysis code end to end. In the current run the simulated difference is *not* significant (p = 0.22), which is exactly the kind of miss an 80%-power test allows 1 time in 5. The simulated lift is an assumption, not a finding.

## The API

```bash
uvicorn api.main:app --reload        # then open http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{
  "tenure_months": 1, "contract": "Month-to-month", "payment_method": "Electronic check",
  "monthly_charges": 29.85, "total_charges": 29.85, "internet_service": "DSL",
  "partner": true, "phone_service": false, "online_backup": "Yes"}'
```

Returns the churn probability, the decision (threshold 0.60), a low / medium / high risk band, the customer's top SHAP reasons in plain English, and a retention plan. With `OPENAI_API_KEY` set the plan is written by an LLM from those reasons; without a key (or if the call fails) it falls back to rules, so the endpoint never breaks.

## Run it yourself

```bash
pip install -r requirements.txt
docker compose up -d db
export DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/churn
./run_pipeline.sh          # SQL -> features -> stats -> models -> segments -> exports
pytest -q                  # API tests
docker compose up --build api
```

Re-running the pipeline gives identical numbers (fixed seeds and a deterministic `ORDER BY` on the training query).

Experiment tracking: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Outputs for business users

- `reports/powerbi/customer_scores.csv`: every customer with churn probability, risk band, segment, top driver and revenue at risk. See `POWERBI.md` for the dashboard build.
- `reports/retention_summary.xlsx`: segment summary, contract × risk table, high-risk drivers, top 500 at-risk customers.
- `reports/segments_pca.png`: the segments in PCA space.

## Project layout

```
sql/          schema (3 tables, indexes) and named analysis queries
src/          load_db, sql_analysis, prepare, stats_tests, train, segment, retention, export_bi
api/          FastAPI service
tests/        API tests
reports/      every result as JSON / CSV / PNG / XLSX
models/       trained pipeline + SHAP explainer
```

## Data

IBM Telco Customer Churn sample dataset (public). 11 brand-new customers had a blank total charge; they were set to 0 because they had not been billed yet, not dropped. The IQR check found no outliers in tenure or charges, so no customers were removed.
