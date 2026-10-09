-- Normalized schema for the telco churn data.
-- One customer row, one account row and one services row per customer,
-- so every business question needs a join, like a real warehouse would.

DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id     VARCHAR(20) PRIMARY KEY,
    gender          VARCHAR(10),
    senior_citizen  BOOLEAN,
    partner         BOOLEAN,
    dependents      BOOLEAN
);

CREATE TABLE accounts (
    customer_id       VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    tenure_months     INTEGER,
    contract          VARCHAR(20),
    paperless_billing BOOLEAN,
    payment_method    VARCHAR(40),
    monthly_charges   NUMERIC(8, 2),
    total_charges     NUMERIC(10, 2),
    churned           BOOLEAN
);

CREATE TABLE services (
    customer_id       VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    phone_service     BOOLEAN,
    multiple_lines    VARCHAR(20),
    internet_service  VARCHAR(20),
    online_security   VARCHAR(20),
    online_backup     VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support      VARCHAR(20),
    streaming_tv      VARCHAR(20),
    streaming_movies  VARCHAR(20)
);

-- The dashboards filter on these columns all the time.
CREATE INDEX idx_accounts_contract ON accounts(contract);
CREATE INDEX idx_accounts_churned  ON accounts(churned);
CREATE INDEX idx_accounts_tenure   ON accounts(tenure_months);
CREATE INDEX idx_services_internet ON services(internet_service);
