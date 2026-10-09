# Power BI dashboard: build steps (about 1–2 hours)

Data: `reports/powerbi/customer_scores.csv` (one row per customer) and `reports/powerbi/segment_summary.csv`.

## 1. Load
Power BI Desktop → Get Data → Text/CSV → `customer_scores.csv` → Load. Repeat for `segment_summary.csv`.
In Model view, relate `customer_scores[segment]` → `segment_summary[segment]` (many-to-one).

## 2. Measures (Home → New measure)
```DAX
Customers = COUNTROWS(customer_scores)
Churn Rate = DIVIDE(SUM(customer_scores[churn]), [Customers])
Avg Churn Risk = AVERAGE(customer_scores[churn_probability])
High Risk Customers = CALCULATE([Customers], customer_scores[risk_band] = "high")
Revenue At Risk = SUM(customer_scores[monthly_revenue_at_risk])
High Risk Share = DIVIDE([High Risk Customers], [Customers])
```

## 3. Page layout
- **Cards:** Customers · Churn Rate · High Risk Customers · Revenue At Risk
- **Clustered bar:** Churn Rate by `contract`
- **Column:** Churn Rate by `tenure_cohort` (sort by cohort)
- **Bar:** High Risk Customers by `top_churn_driver`
- **Matrix:** rows `segment`, values Customers, Churn Rate, Avg Churn Risk, Revenue At Risk
- **Table:** top at-risk customers (customer_id, churn_probability, top_churn_driver, monthly_charges), sorted by probability
- **Slicers:** `risk_band`, `internet_service`, `payment_method`

## 4. Publish
Save as `reports/churn_dashboard.pbix` and commit it. If you have a Power BI service account, Publish → "Publish to web" gives a public link for the resume; otherwise add screenshots to the README.
