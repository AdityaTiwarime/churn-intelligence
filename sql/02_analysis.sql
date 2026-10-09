-- name: overall_churn
SELECT COUNT(*)                                          AS customers,
       SUM(CASE WHEN churned THEN 1 ELSE 0 END)          AS churned,
       ROUND(100.0 * AVG(CASE WHEN churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM accounts;

-- name: churn_by_contract
SELECT a.contract,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN a.churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct,
       ROUND(AVG(a.monthly_charges), 2) AS avg_monthly_charges
FROM accounts a
GROUP BY a.contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_cohort
WITH cohorts AS (
    SELECT customer_id,
           churned,
           CASE
               WHEN tenure_months <= 6  THEN '01: 0-6 months'
               WHEN tenure_months <= 12 THEN '02: 7-12 months'
               WHEN tenure_months <= 24 THEN '03: 13-24 months'
               WHEN tenure_months <= 48 THEN '04: 25-48 months'
               ELSE '05: 49+ months'
           END AS tenure_cohort
    FROM accounts
)
SELECT tenure_cohort,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM cohorts
GROUP BY tenure_cohort
ORDER BY tenure_cohort;

-- name: churn_by_internet_and_support
SELECT s.internet_service,
       s.tech_support,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN a.churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM services s
JOIN accounts a ON a.customer_id = s.customer_id
WHERE s.internet_service <> 'No'
GROUP BY s.internet_service, s.tech_support
ORDER BY churn_rate_pct DESC;

-- name: churn_by_payment_method
SELECT a.payment_method,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN a.churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM accounts a
GROUP BY a.payment_method
ORDER BY churn_rate_pct DESC;

-- name: senior_churn
SELECT c.senior_citizen,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN a.churned THEN 1 ELSE 0 END), 2) AS churn_rate_pct
FROM customers c
JOIN accounts a ON a.customer_id = c.customer_id
GROUP BY c.senior_citizen;

-- name: revenue_at_risk
-- Monthly revenue walking out the door, and how it ranks inside each contract type.
WITH churned_revenue AS (
    SELECT a.contract,
           s.internet_service,
           SUM(a.monthly_charges) AS lost_monthly_revenue,
           COUNT(*)               AS churned_customers
    FROM accounts a
    JOIN services s ON s.customer_id = a.customer_id
    WHERE a.churned
    GROUP BY a.contract, s.internet_service
)
SELECT contract,
       internet_service,
       churned_customers,
       lost_monthly_revenue,
       RANK() OVER (PARTITION BY contract ORDER BY lost_monthly_revenue DESC) AS rank_in_contract,
       ROUND(100.0 * lost_monthly_revenue / SUM(lost_monthly_revenue) OVER (), 2) AS share_of_total_pct
FROM churned_revenue
ORDER BY lost_monthly_revenue DESC;
