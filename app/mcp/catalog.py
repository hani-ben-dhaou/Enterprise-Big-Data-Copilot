"""
Catalog services — metadata access for the MCP layer.

Two implementations share one stable contract (``CatalogService``):

  * ``InMemoryCatalog`` — curated demo catalog (Hive/Iceberg sales tables).
    Kept as the default for offline development and deterministic tests.
    No row data is stored, so sample/statistic tools report "no data".
  * ``TrinoCatalog``   — live metadata via Trino ``information_schema``.
    Used when the metadata source is configured to ``trino``.

``get_catalog_service()`` is the DI seam the MCP server and client use,
so the metadata backend is swappable without touching tool definitions.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.trino import TrinoService

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Catalog definition — structured as catalog → schema → table → columns
# ---------------------------------------------------------------------------

_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "hive": {
        "sales": {
            "orders": {
                "comment": "Customer orders fact table (Iceberg format)",
                "columns": [
                    {"name": "order_id",        "data_type": "BIGINT",    "nullable": False, "comment": "Unique order identifier"},
                    {"name": "customer_id",     "data_type": "BIGINT",    "nullable": False, "comment": "FK to customers.customer_id"},
                    {"name": "order_date",      "data_type": "DATE",      "nullable": False, "comment": "Date the order was placed"},
                    {"name": "total_amount",    "data_type": "DECIMAL(15,2)", "nullable": True, "comment": "Total order value in USD"},
                    {"name": "status",          "data_type": "VARCHAR",   "nullable": True,  "comment": "Order status: PENDING/SHIPPED/DELIVERED/CANCELLED"},
                    {"name": "region",          "data_type": "VARCHAR",   "nullable": True,  "comment": "Geographic sales region"},
                    {"name": "product_id",      "data_type": "BIGINT",    "nullable": True,  "comment": "FK to products.product_id"},
                    {"name": "quantity",        "data_type": "INTEGER",   "nullable": True,  "comment": "Number of units ordered"},
                ],
            },
            "customers": {
                "comment": "Customer dimension table",
                "columns": [
                    {"name": "customer_id",     "data_type": "BIGINT",    "nullable": False, "comment": "Unique customer identifier"},
                    {"name": "name",            "data_type": "VARCHAR",   "nullable": False, "comment": "Full customer name"},
                    {"name": "email",           "data_type": "VARCHAR",   "nullable": True,  "comment": "Contact email"},
                    {"name": "country",         "data_type": "VARCHAR",   "nullable": True,  "comment": "Country of residence"},
                    {"name": "created_at",      "data_type": "TIMESTAMP", "nullable": False, "comment": "Account creation timestamp"},
                    {"name": "segment",         "data_type": "VARCHAR",   "nullable": True,  "comment": "Customer segment: ENTERPRISE/SMB/CONSUMER"},
                ],
            },
            "products": {
                "comment": "Product dimension table",
                "columns": [
                    {"name": "product_id",      "data_type": "BIGINT",    "nullable": False, "comment": "Unique product identifier"},
                    {"name": "name",            "data_type": "VARCHAR",   "nullable": False, "comment": "Product name"},
                    {"name": "category",        "data_type": "VARCHAR",   "nullable": True,  "comment": "Product category"},
                    {"name": "unit_price",      "data_type": "DECIMAL(10,2)", "nullable": True, "comment": "List price per unit"},
                    {"name": "supplier_id",     "data_type": "BIGINT",    "nullable": True,  "comment": "FK to suppliers.supplier_id"},
                ],
            },
        },
        "analytics": {
            "revenue_daily": {
                "comment": "Daily revenue aggregate (partitioned by date)",
                "columns": [
                    {"name": "report_date",     "data_type": "DATE",      "nullable": False, "comment": "Partition key — reporting date"},
                    {"name": "region",          "data_type": "VARCHAR",   "nullable": False, "comment": "Sales region"},
                    {"name": "product_category","data_type": "VARCHAR",   "nullable": True,  "comment": "Product category"},
                    {"name": "total_revenue",   "data_type": "DECIMAL(18,2)", "nullable": True, "comment": "Sum of revenue for the day"},
                    {"name": "order_count",     "data_type": "BIGINT",    "nullable": True,  "comment": "Number of orders"},
                    {"name": "avg_order_value", "data_type": "DECIMAL(10,2)", "nullable": True, "comment": "Average order value"},
                ],
            },
            "customer_metrics": {
                "comment": "Customer-level KPI aggregates",
                "columns": [
                    {"name": "customer_id",     "data_type": "BIGINT",    "nullable": False, "comment": "Customer identifier"},
                    {"name": "lifetime_value",  "data_type": "DECIMAL(15,2)", "nullable": True, "comment": "Total spend to date"},
                    {"name": "order_count",     "data_type": "BIGINT",    "nullable": True,  "comment": "Total number of orders"},
                    {"name": "last_order_date", "data_type": "DATE",      "nullable": True,  "comment": "Most recent order date"},
                    {"name": "churn_risk_score","data_type": "DOUBLE",    "nullable": True,  "comment": "ML churn risk 0-1"},
                ],
            },
        },
        "raw": {
            "events": {
                "comment": "Raw clickstream events (Iceberg, partitioned by event_date)",
                "columns": [
                    {"name": "event_id",        "data_type": "VARCHAR",   "nullable": False, "comment": "UUID event identifier"},
                    {"name": "event_date",      "data_type": "DATE",      "nullable": False, "comment": "Partition key"},
                    {"name": "event_type",      "data_type": "VARCHAR",   "nullable": False, "comment": "Type: PAGE_VIEW/CLICK/ADD_TO_CART/PURCHASE"},
                    {"name": "user_id",         "data_type": "BIGINT",    "nullable": True,  "comment": "Nullable for anonymous sessions"},
                    {"name": "session_id",      "data_type": "VARCHAR",   "nullable": True,  "comment": "Browser session"},
                    {"name": "properties",      "data_type": "JSON",      "nullable": True,  "comment": "Event payload as JSON"},
                    {"name": "created_at",      "data_type": "TIMESTAMP WITH TIME ZONE", "nullable": False, "comment": "Event ingestion timestamp"},
                ],
            },
        },
    },
}

# Matches column comments declaring a foreign key, e.g.
#   "FK to customers.customer_id"       → (customers, customer_id)
#   "FK -> products.product_id"         → (products, product_id)
#   "References products(product_id)"   → (products, product_id)
_FK_RE = re.compile(
    r"(?:(?:\bfk|foreign\s+key)\s+(?:to|->|=>)\s+|references\s+)"
    r"([a-z_][a-z0-9_]*)"
    r"(?:\.([a-z_][a-z0-9_]*)|\(\s*([a-z_][a-z0-9_]*)\s*\))?",
    re.IGNORECASE,
)


def _parse_foreign_keys(columns: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract FK relationships from column comments (best effort)."""
    relationships: list[dict[str, str]] = []
    for col in columns:
        comment = (col.get("comment") or "").strip()
        if not comment:
            continue
        for match in _FK_RE.finditer(comment):
            table = match.group(1)
            target_col = match.group(2) or match.group(3) or "id"
            relationships.append({
                "from_column": col["name"],
                "to_table": table,
                "to_column": target_col,
                "type": "foreign_key",
            })
    return relationships


class CatalogService(Protocol):
    """The stable metadata contract consumed by the MCP tool layer."""

    def list_catalogs(self) -> list[str]: ...

    def list_schemas(self, catalog: str) -> list[str]: ...

    def list_tables(self, catalog: str, schema: str) -> list[str]: ...

    def describe_table(self, catalog: str, schema: str, table: str) -> dict: ...

    def search_tables(self, keyword: str) -> list[dict]: ...


class InMemoryCatalog:
    """
    In-memory implementation of the metadata catalog (offline / demo).

    Contract methods must be preserved when swapping to a real backend.
    """

    def list_catalogs(self) -> list[str]:
        return list(_CATALOG.keys())

    def list_schemas(self, catalog: str) -> list[str]:
        if catalog not in _CATALOG:
            return []
        return list(_CATALOG[catalog].keys())

    def list_tables(self, catalog: str, schema: str) -> list[str]:
        if catalog not in _CATALOG or schema not in _CATALOG[catalog]:
            return []
        return list(_CATALOG[catalog][schema].keys())

    def describe_table(self, catalog: str, schema: str, table: str) -> dict:
        """Return column definitions and metadata for a single table."""
        try:
            meta = _CATALOG[catalog][schema][table]
        except KeyError:
            return {"error": f"Table not found: {catalog}.{schema}.{table}"}

        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "comment": meta.get("comment", ""),
            "columns": meta["columns"],
            "fully_qualified_name": f"{catalog}.{schema}.{table}",
        }

    def search_tables(self, keyword: str) -> list[dict]:
        """Search table + column names for a keyword (case-insensitive)."""
        kw = keyword.lower()
        results: list[dict] = []

        for cat, schemas in _CATALOG.items():
            for schema, tables in schemas.items():
                for table, meta in tables.items():
                    fqn = f"{cat}.{schema}.{table}"
                    matched_columns = [
                        col["name"]
                        for col in meta["columns"]
                        if kw in col["name"].lower()
                    ]
                    if kw in table.lower() or matched_columns:
                        results.append({
                            "fully_qualified_name": fqn,
                            "table": table,
                            "schema": schema,
                            "catalog": cat,
                            "comment": meta.get("comment", ""),
                            "matched_columns": matched_columns,
                        })

        return results


class TrinoCatalog:
    """
    Live metadata catalog backed by Trino ``information_schema``.

    Mirrors the InMemoryCatalog return shapes so the MCP tools and the
    client do not care which backend is in use.
    """

    def __init__(self, service: TrinoService | None = None) -> None:
        self._service = service or TrinoService()

    def list_catalogs(self) -> list[str]:
        return self._service.list_catalogs()

    def list_schemas(self, catalog: str) -> list[str]:
        return self._service.list_schemas(catalog)

    def list_tables(self, catalog: str, schema: str) -> list[str]:
        return self._service.list_tables(catalog, schema)

    def describe_table(self, catalog: str, schema: str, table: str) -> dict:
        try:
            columns = self._service.get_columns(catalog, schema, table)
        except Exception as exc:
            return {"error": f"Could not describe {catalog}.{schema}.{table}: {exc}"}

        if not columns:
            return {"error": f"Table not found: {catalog}.{schema}.{table}"}

        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "comment": "",
            "columns": columns,
            "fully_qualified_name": f"{catalog}.{schema}.{table}",
        }

    def search_tables(self, keyword: str, max_results: int = 20) -> list[dict]:
        """Search table and column names across all known objects."""
        kw = keyword.lower()
        results: list[dict] = []

        for cat in self.list_catalogs():
            for schema in self.list_schemas(cat):
                if schema.lower() == "information_schema":
                    continue
                for table in self.list_tables(cat, schema):
                    meta = self.describe_table(cat, schema, table)
                    matched_columns = [
                        col["name"]
                        for col in meta.get("columns", [])
                        if kw in col["name"].lower()
                    ]
                    if kw in table.lower() or matched_columns:
                        results.append({
                            "fully_qualified_name": f"{cat}.{schema}.{table}",
                            "table": table,
                            "schema": schema,
                            "catalog": cat,
                            "comment": meta.get("comment", ""),
                            "matched_columns": matched_columns,
                        })
                        if len(results) >= max_results:
                            return results
        return results


def get_catalog_service() -> CatalogService:
    """
    DI factory: return the configured metadata backend.

    * ``inmemory`` (default) → InMemoryCatalog (offline, deterministic)
    * ``trino``            → TrinoCatalog (live information_schema)
    """
    source = settings.mcp_metadata_source
    if source == "trino":
        try:
            service = TrinoCatalog()
            logger.info("catalog_service_initialized", source="trino")
            return service
        except Exception as exc:
            logger.warning("trino_catalog_unavailable_falling_back", error=str(exc))

    logger.info("catalog_service_initialized", source="inmemory")
    return InMemoryCatalog()