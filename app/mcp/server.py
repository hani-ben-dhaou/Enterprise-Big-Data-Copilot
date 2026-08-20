"""
MCP Server — exposes Big Data platform capabilities to AI agents via MCP.

The server is the *only* layer that talks to the data platform.

    LLM Agent
        ↓
    MCP Client (app/mcp/client.py)
        ↓
    MCP Server  (this module)   ← FastMCP
        ↓
    Catalog / Trino services
        ↓
    Trino

Tools:
  - Metadata : list_catalogs, list_schemas, list_tables, describe_table,
               get_columns, search_tables
  - Query    : validate_sql, execute_sql
  - Profiling: get_table_sample, get_table_statistics, get_table_relationships

Run with:
    python -m app.mcp.server                 # SSE on MCP_SERVER_PORT (default 8001)
    python -m app.mcp.server --transport stdio
"""

from __future__ import annotations

import argparse
from typing import Any

from fastmcp import FastMCP

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import SchemaContext
from app.mcp.catalog import (
    CatalogService,
    _parse_foreign_keys,
    get_catalog_service,
)
from app.services.trino import TrinoService
from app.validator.sql_validator import SQLValidator

logger = get_logger(__name__)
settings = get_settings()

mcp = FastMCP(
    "bigdata-copilot",
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
    log_level=settings.log_level,
)

_catalog: CatalogService | None = None
_trino: TrinoService | None = None
_validator = SQLValidator()


def _get_catalog() -> CatalogService:
    global _catalog
    if _catalog is None:
        _catalog = get_catalog_service()
    return _catalog


def _get_trino() -> TrinoService:
    global _trino
    if _trino is None:
        _trino = TrinoService()
    return _trino


def _live_platform() -> bool:
    """True when the configured metadata source is backed by Trino."""
    return settings.mcp_metadata_source == "trino"


# ---------------------------------------------------------------------------
# Metadata tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_catalogs() -> dict[str, Any]:
    """List all available data catalogs (e.g. hive, tpch, system)."""
    return {"catalogs": _get_catalog().list_catalogs()}


@mcp.tool()
def list_schemas(catalog: str) -> dict[str, Any]:
    """List all schemas within a catalog."""
    return {
        "catalog": catalog,
        "schemas": _get_catalog().list_schemas(catalog),
    }


@mcp.tool()
def list_tables(catalog: str, schema: str) -> dict[str, Any]:
    """List all tables within a catalog.schema."""
    return {
        "catalog": catalog,
        "schema": schema,
        "tables": _get_catalog().list_tables(catalog, schema),
    }


@mcp.tool()
def describe_table(catalog: str, schema: str, table: str) -> dict[str, Any]:
    """Return full metadata (comment + column definitions) for one table."""
    return _get_catalog().describe_table(catalog, schema, table)


@mcp.tool()
def get_columns(catalog: str, schema: str, table: str) -> dict[str, Any]:
    """Return the columns (name, type, nullable, comment) of a table."""
    meta = _get_catalog().describe_table(catalog, schema, table)
    if "error" in meta:
        return meta
    return {"columns": meta["columns"]}


@mcp.tool()
def search_tables(query: str) -> dict[str, Any]:
    """Search tables by table or column name; returns matching tables."""
    results = _get_catalog().search_tables(query)
    return {"query": query, "count": len(results), "results": results}


# ---------------------------------------------------------------------------
# Query tools
# ---------------------------------------------------------------------------


@mcp.tool()
def validate_sql(sql: str) -> dict[str, Any]:
    """
    Validate a SQL statement without executing it.
    Enforces read-only SELECT rules and Trino parseability.
    """
    result = _validator.validate(sql, SchemaContext())
    return result.model_dump()


@mcp.tool()
def execute_sql(sql: str, limit: int | None = None) -> dict[str, Any]:
    """
    Execute a read-only SELECT against the data platform and return rows.
    Rejects any non-SELECT statement before execution.
    """
    if not settings.mcp_query_execution_enabled:
        return {"error": "Query execution is disabled on this server."}

    validation = _validator.validate(sql, SchemaContext())
    if not validation.is_valid:
        return {
            "error": "Query rejected before execution.",
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

    cap = limit if limit is not None else settings.mcp_default_limit
    try:
        return _get_trino().execute_query(sql, limit=cap)
    except Exception as exc:
        logger.error("execute_sql_failed", error=str(exc))
        return {"error": f"Query execution failed: {exc}"}


# ---------------------------------------------------------------------------
# Profiling tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_table_sample(
    catalog: str, schema: str, table: str, limit: int = 5
) -> dict[str, Any]:
    """Return a small sample of rows for a table (preview only)."""
    meta = _get_catalog().describe_table(catalog, schema, table)
    if "error" in meta:
        return meta
    if not settings.mcp_profiling_enabled:
        return {"error": "Table profiling is disabled on this server."}
    if not _live_platform():
        return {
            "error": "No row data available: the metadata source is not a queryable "
            "Trino platform (set MCP_METADATA_SOURCE=trino)."
        }
    try:
        return _get_trino().get_table_sample(catalog, schema, table, limit=limit)
    except Exception as exc:
        logger.error("table_sample_failed", error=str(exc))
        return {"error": f"Could not sample {catalog}.{schema}.{table}: {exc}"}


@mcp.tool()
def get_table_statistics(catalog: str, schema: str, table: str) -> dict[str, Any]:
    """Return schema-level statistics; row-count estimate when on Trino."""
    meta = _get_catalog().describe_table(catalog, schema, table)
    if "error" in meta:
        return meta

    columns = meta["columns"]
    result: dict[str, Any] = {
        "catalog": catalog,
        "schema": schema,
        "table": table,
        "column_count": len(columns),
        "nullable_columns": [
            c["name"] for c in columns if c.get("nullable", True)
        ],
        "not_null_columns": [
            c["name"] for c in columns if not c.get("nullable", True)
        ],
        "row_count_estimate": None,
        "estimate_available": False,
    }

    if _live_platform() and settings.mcp_profiling_enabled:
        try:
            stats = _get_trino().get_table_statistics(catalog, schema, table)
            result["row_count_estimate"] = stats.get("row_count_estimate")
            result["estimate_available"] = bool(stats.get("estimate_available"))
        except Exception as exc:
            logger.warning("table_stats_failed", error=str(exc))

    return result


@mcp.tool()
def get_table_relationships(catalog: str, schema: str, table: str) -> dict[str, Any]:
    """Return foreign-key relationships declared on a table's columns."""
    meta = _get_catalog().describe_table(catalog, schema, table)
    if "error" in meta:
        return meta

    relationships = _parse_foreign_keys(meta["columns"])
    return {
        "catalog": catalog,
        "schema": schema,
        "table": table,
        "relationship_count": len(relationships),
        "relationships": relationships,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise Big Data Copilot MCP server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="MCP transport (default: sse on mcp_server_port)",
    )
    args = parser.parse_args()

    logger.info(
        "mcp_server_starting",
        transport=args.transport,
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()