# Trino SQL Reference — V1 Documentation

## Overview

Trino (formerly PrestoSQL) is a distributed SQL query engine designed for fast
analytic queries over large datasets. It supports ANSI SQL with extensions.

---

## Date and Time Functions

### DATE_TRUNC
Truncates a timestamp to the specified unit.

```sql
DATE_TRUNC('month', order_date)      -- first day of the month
DATE_TRUNC('year',  created_at)      -- first day of the year
DATE_TRUNC('week',  event_date)      -- Monday of the week (ISO)
```

### DATE_ADD / DATE_DIFF
```sql
-- Add 30 days to a date
DATE_ADD('day', 30, order_date)

-- Difference in days between two dates
DATE_DIFF('day', start_date, end_date)
```

### Current Date/Time
```sql
CURRENT_DATE          -- today's date
CURRENT_TIMESTAMP     -- current timestamp with time zone
NOW()                 -- alias for CURRENT_TIMESTAMP
```

---

## Window Functions

Window functions compute values across a set of rows related to the current row.

```sql
-- Running total
SUM(revenue) OVER (
    PARTITION BY region
    ORDER BY report_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS running_revenue

-- Row number per partition
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn

-- Rank with gaps
RANK() OVER (ORDER BY total_amount DESC) AS rank

-- Lag (previous row value)
LAG(revenue, 1) OVER (PARTITION BY region ORDER BY report_date) AS prev_revenue
```

---

## Approximate Aggregates (for large datasets)

```sql
-- Approx distinct count (faster than COUNT(DISTINCT ...))
APPROX_DISTINCT(customer_id)

-- Approx percentile
APPROX_PERCENTILE(total_amount, 0.5)    -- median
APPROX_PERCENTILE(total_amount, 0.95)   -- 95th percentile
```

---

## JSON Functions

```sql
-- Extract a field from a JSON column
JSON_EXTRACT_SCALAR(properties, '$.event_type')

-- Check JSON field existence
JSON_EXTRACT(properties, '$.user_id') IS NOT NULL
```

---

## Safe Casting

Always use TRY_CAST when casting user-generated or nullable data.

```sql
TRY_CAST(user_input AS BIGINT)     -- returns NULL on failure
TRY_CAST(amount_str AS DECIMAL(10,2))
```

---

## CTEs (Common Table Expressions)

Use CTEs to break complex queries into readable steps.

```sql
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date)  AS month,
        region,
        SUM(total_amount)                AS revenue
    FROM hive.sales.orders
    WHERE status = 'DELIVERED'
    GROUP BY 1, 2
),
ranked AS (
    SELECT
        *,
        RANK() OVER (PARTITION BY month ORDER BY revenue DESC) AS rk
    FROM monthly_revenue
)
SELECT * FROM ranked WHERE rk <= 5
LIMIT 100
```

---

## Best Practices for Big Data Queries

1. **Always filter on partition columns** — reduces data scanned dramatically.
2. **Use APPROX functions** for exploratory queries on tables with billions of rows.
3. **Avoid SELECT \*** — list columns explicitly for performance.
4. **Push filters early** — filter in CTEs or subqueries before joining.
5. **Use LIMIT** — always add a LIMIT for exploratory queries.
6. **Prefer integer joins** — join on BIGINT keys, not VARCHAR.