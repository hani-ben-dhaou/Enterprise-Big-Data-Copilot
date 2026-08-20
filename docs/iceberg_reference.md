# Apache Iceberg — Trino Integration Reference

## What is Iceberg?

Apache Iceberg is an open table format for huge analytic datasets.
It brings ACID transactions, schema evolution, and time-travel to data lakes.

---

## Iceberg + Trino Syntax

### Time Travel Queries

```sql
-- Query a snapshot by timestamp
SELECT * FROM hive.sales.orders
FOR TIMESTAMP AS OF TIMESTAMP '2024-01-01 00:00:00 UTC'
LIMIT 100

-- Query a snapshot by snapshot ID
SELECT * FROM hive.sales.orders
FOR VERSION AS OF 123456789
LIMIT 100
```

### Partition Pruning

Always filter on partition columns for Iceberg tables.

```sql
-- GOOD: filters on partition column (event_date)
SELECT COUNT(*) FROM hive.raw.events
WHERE event_date BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'

-- BAD: no partition filter — full table scan
SELECT COUNT(*) FROM hive.raw.events
WHERE event_type = 'PURCHASE'
```

### Schema Evolution

Iceberg supports adding and dropping columns without rewriting data.
When querying older snapshots, missing columns return NULL.

---

## Table Metadata Views

```sql
-- List snapshots
SELECT * FROM hive.sales."orders$snapshots"

-- List partitions
SELECT * FROM hive.sales."orders$partitions"

-- List files
SELECT * FROM hive.sales."orders$files"
```

---

## Key Iceberg Concepts

- **Snapshot**: immutable point-in-time view of the table
- **Manifest**: tracks data file locations and statistics
- **Partition spec**: defines how data is physically organized
- **Hidden partitioning**: Iceberg partitions by value; no need to add partition cols to queries

---

## Common Patterns

### Incremental Processing

```sql
-- Get records added since a checkpoint
SELECT *
FROM hive.raw.events
WHERE created_at > TIMESTAMP '2024-06-01 00:00:00 UTC'
  AND event_date >= DATE '2024-06-01'
ORDER BY created_at
LIMIT 10000
```

### Merge Pattern (V2 — not supported in V1 read-only mode)

In V1 this system generates SELECT-only queries.
INSERT/MERGE operations are V2 scope.