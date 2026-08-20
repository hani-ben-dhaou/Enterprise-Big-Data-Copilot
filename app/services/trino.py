"""
TrinoService — thin, safe wrapper around the Trino client (DB-API).

Responsibilities:
  - Execute read-only SELECT queries and return (columns, rows).
  - Introspect ``information_schema`` for dynamic catalog metadata.

Design rules:
  - The MCP layer calls this service; the SQL agent never talks to Trino
    directly.
  - Every identifier used in generated SQL is quoted to prevent injection.
  - Only SELECT statements are allowed; safety enforcement lives in the
    SQLValidator, but this service refuses runaway fetches via a hard limit.
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

try:
    import trino
except ImportError:  # pragma: no cover
    trino = None


def quote_identifier(name: str) -> str:
    """Quote a Trino identifier (double quotes, escaped)."""
    return '"' + str(name).replace('"', '""') + '"'


class TrinoService:
    """
    Stateless Trino connection wrapper. Connections are cheap over HTTP,
    so we open one per call and close it afterwards.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        if trino is None:  # pragma: no cover
            raise RuntimeError(
                "The 'trino' package is not installed. Run: pip install trino tzlocal"
            )

        self._host = host or settings.trino_host
        self._port = port or settings.trino_port
        self._user = user or settings.trino_user
        self._catalog = catalog or settings.trino_catalog
        self._schema = schema or settings.trino_schema

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(
        self,
        sql: str,
        limit: int | None = None,
        hard_cap: int = 10_000,
    ) -> dict[str, Any]:
        """
        Execute a SELECT query and return columns + (capped) rows.

        Args:
            sql: The SQL to execute. Callers are responsible for validating
                that it is a read-only SELECT before calling.
            limit: Maximum number of rows to fetch. None fetches everything.
            hard_cap: Upper bound on limit, protects against runaway fetches.

        Returns:
            {
                "catalog": ...,
                "schema": ...,
                "columns": [str],
                "rows": [[...]],
                "row_count": int,
                "truncated": bool,
            }
        """
        if limit is not None:
            limit = max(1, min(int(limit), hard_cap))

        conn = self._connection()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            columns = [desc[0] for desc in (cur.description or [])]
            rows = cur.fetchmany(limit) if limit is not None else cur.fetchall()
        except Exception as exc:
            logger.error("trino_execution_failed", error=str(exc))
            raise
        finally:
            conn.close()

        return {
            "catalog": self._catalog,
            "schema": self._schema,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": limit is not None and len(rows) >= limit,
        }

    # ------------------------------------------------------------------
    # Catalog introspection
    # ------------------------------------------------------------------

    def list_catalogs(self) -> list[str]:
        # Trino does not expose a `catalogs` table in information_schema
        # (only columns/tables/views/schemata); SHOW CATALOGS is the supported
        # way to enumerate the available catalogs.
        rows = self._query("SHOW CATALOGS")
        return self._rows_to_list(rows)

    def list_schemas(self, catalog: str) -> list[str]:
        rows = self._query(
            f"SELECT schema_name FROM {quote_identifier(catalog)}.information_schema.schemata "
            "ORDER BY schema_name"
        )
        return self._rows_to_list(rows)

    def list_tables(self, catalog: str, schema: str) -> list[str]:
        rows = self._query(
            f"SELECT table_name FROM {quote_identifier(catalog)}.information_schema.tables "
            "WHERE table_schema = ? AND table_type IN ('BASE TABLE', 'VIEW') "
            "ORDER BY table_name",
            params=[schema],
        )
        return self._rows_to_list(rows)

    def get_columns(self, catalog: str, schema: str, table: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT column_name, data_type, is_nullable, comment "
            f"FROM {quote_identifier(catalog)}.information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position",
            params=[schema, table],
        )
        columns: list[dict[str, Any]] = []
        for row in rows:
            columns.append({
                "name": row[0],
                "data_type": row[1],
                "nullable": str(row[2]).upper() != "NO",
                "comment": row[3],
            })
        return columns

    # ------------------------------------------------------------------
    # Data profiling helpers
    # ------------------------------------------------------------------

    def get_table_sample(
        self, catalog: str, schema: str, table: str, limit: int = 5
    ) -> dict[str, Any]:
        """Return a sample of rows from a table (read-only SELECT)."""
        limit = max(1, min(int(limit), 1000))
        sql = (
            f"SELECT * FROM {quote_identifier(catalog)}.{quote_identifier(schema)}."
            f"{quote_identifier(table)} LIMIT {limit}"
        )
        return self.execute_query(sql, limit=limit)

    def get_table_statistics(
        self, catalog: str, schema: str, table: str
    ) -> dict[str, Any]:
        """
        Estimated table statistics via ``SHOW STATS``.
        Returns a best-effort result; never raises on unsupported catalogs.
        """
        try:
            rows = self._query(
                f"SHOW STATS FOR {quote_identifier(catalog)}.{quote_identifier(schema)}."
                f"{quote_identifier(table)}"
            )
        except Exception as exc:
            logger.warning("show_stats_failed", error=str(exc))
            return {"estimate_available": False, "error": str(exc)}

        # SHOW STATS returns one row per column plus a summary row whose
        # column_name is NULL and 'row_count' sits at index 4:
        # (column_name, data_size, distinct_values_count, nulls_fraction,
        #  row_count, low_value, high_value)
        row_counts = [r for r in rows if r and r[0] is None and len(r) > 4]
        estimate = row_counts[0][4] if row_counts else None

        columns = self.get_columns(catalog, schema, table)
        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "estimate_available": True,
            "row_count_estimate": estimate,
            "column_count": len(columns),
            "columns": [c["name"] for c in columns],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connection(self):
        if trino is None:  # pragma: no cover
            raise RuntimeError("trino package is not installed")
        return trino.dbapi.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            catalog=self._catalog,
            schema=self._schema,
            http_scheme="http",
        )

    def _query(self, sql: str, params: list[Any] | None = None) -> list[list[Any]]:
        conn = self._connection()
        try:
            cur = conn.cursor()
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            return [list(r) for r in (cur.fetchall() or [])]
        finally:
            conn.close()

    @staticmethod
    def _rows_to_list(rows: list[list[Any]]) -> list[str]:
        return [str(r[0]) for r in rows]