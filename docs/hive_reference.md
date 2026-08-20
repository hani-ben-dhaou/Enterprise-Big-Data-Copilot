# Apache Hive — SQL Reference for Trino Users

## Overview

Apache Hive provides the metastore layer (HMS) used by Trino as a catalog backend.
In this system, Hive tables are queried through Trino — not via HiveQL directly.

---

## Data Types Mapping (Hive → Trino)

| Hive Type        | Trino Type               | Notes                          |
|------------------|--------------------------|--------------------------------|
| STRING           | VARCHAR                  | unbounded string               |
| INT              | INTEGER                  |                                |
| BIGINT           | BIGINT                   |                                |
| FLOAT            | REAL                     |                                |
| DOUBLE           | DOUBLE                   |                                |
| DECIMAL(p,s)     | DECIMAL(p,s)             |                                |
| BOOLEAN          | BOOLEAN                  |                                |
| TIMESTAMP        | TIMESTAMP                | naive (no TZ)                  |
| DATE             | DATE                     |                                |
| BINARY           | VARBINARY                |                                |
| ARRAY<T>         | ARRAY(T)                 |                                |
| MAP<K,V>         | MAP(K,V)                 |                                |
| STRUCT<...>      | ROW(...)                 |                                |

---

## Partition Awareness

Hive tables are often partitioned. Always include partition columns in WHERE clauses.

```sql
-- Partitioned by region and order_date
SELECT order_id, total_amount
FROM hive.sales.orders
WHERE region = 'EMEA'
  AND order_date >= DATE '2024-01-01'
LIMIT 1000
```

Not filtering on partition columns causes full table scans (very expensive).

---

## ARRAY and MAP Access in Trino

```sql
-- Access first element of an array column
SELECT product_tags[1] FROM hive.sales.products LIMIT 10

-- Access a map key
SELECT properties['event_type'] FROM hive.raw.events LIMIT 10

-- Explode an array (UNNEST in Trino)
SELECT o.order_id, tag
FROM hive.sales.orders o
CROSS JOIN UNNEST(o.tags) AS t(tag)
LIMIT 100
```

---

## Hive Metastore Queries via Trino

```sql
-- List all tables in a schema
SELECT table_name
FROM hive.information_schema.tables
WHERE table_schema = 'sales'

-- List all columns
SELECT column_name, data_type, is_nullable
FROM hive.information_schema.columns
WHERE table_schema = 'sales'
  AND table_name   = 'orders'
ORDER BY ordinal_position
```

---

## File Formats

| Format    | Notes                                              |
|-----------|----------------------------------------------------|
| ORC       | Best for Hive; columnar, compressed                |
| Parquet   | Best for Iceberg; columnar, widely supported       |
| Avro      | Row-based; good for streaming ingestion            |
| Text/CSV  | Avoid for analytics — no predicate pushdown        |

Trino reads all formats transparently; no syntax change needed.