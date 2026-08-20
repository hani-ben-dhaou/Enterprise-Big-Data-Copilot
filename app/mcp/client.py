"""
MCP Client — the agent-facing façade over MCP tools.

The public API mirrors the MCP server tools 1:1 and always returns
JSON-serializable dicts, so callers do not care which transport is used.

Transports:
  * ``sse``       — real Model Context Protocol: connects to the FastMCP
                    server (app/mcp/server.py) over SSE using the official
                    `mcp` python SDK.
  * ``inprocess`` — offline fallback that dispatches tool calls to the
                    local catalog / validator / Trino services directly.
                    Used for development and deterministic tests.

The orchestrator (node_retrieve_schema) keeps using ``get_schema_context``
unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import MCPError
from app.core.logging import get_logger
from app.core.models import ColumnInfo, SchemaContext, TableSchema
from app.mcp.catalog import get_catalog_service
from app.validator.sql_validator import SQLValidator

logger = get_logger(__name__)
settings = get_settings()

_MCP_SERVER_URL = f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"


# ---------------------------------------------------------------------------
# Remote (SSE) transport helpers — official mcp python SDK
# ---------------------------------------------------------------------------

def _call_remote(tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Call a tool on the remote MCP server over SSE."""
    import asyncio
    import json

    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as exc:  # pragma: no cover
        return {
            "error": (
                "SSE transport is enabled but the 'mcp' python SDK is not "
                "installed. Add 'mcp' to requirements.txt or set "
                "MCP_TRANSPORT=inprocess."
            ),
            "detail": str(exc),
        }

    async def _run() -> dict[str, Any]:
        async with sse_client(_MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name, arguments=arguments or {}
                )
                if getattr(result, "isError", False):
                    return {"error": _content_to_text(result)}
                structured = getattr(result, "structuredContent", None)
                if structured is not None:
                    return structured
                text = _content_to_text(result)
                if not text:
                    return {}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"content": text}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.warning("mcp_remote_call_failed", tool=tool_name, error=str(exc))
        return {"error": f"Remote MCP tool '{tool_name}' failed: {exc}"}


def _content_to_text(result: Any) -> str:
    """Extract text from an MCP CallToolResult's content blocks."""
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# In-process transport — offline dispatch to local services
# ---------------------------------------------------------------------------

_validator = SQLValidator()
_local_catalog = get_catalog_service()


def _call_local(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool locally (no network), mirroring server semantics."""
    catalog = _local_catalog

    if tool_name == "list_catalogs":
        return {"catalogs": catalog.list_catalogs()}
    if tool_name == "list_schemas":
        cat = arguments.get("catalog", "")
        return {"catalog": cat, "schemas": catalog.list_schemas(cat)}
    if tool_name == "list_tables":
        cat, schema = arguments.get("catalog", ""), arguments.get("schema", "")
        return {"catalog": cat, "schema": schema, "tables": catalog.list_tables(cat, schema)}
    if tool_name == "describe_table":
        return catalog.describe_table(
            arguments.get("catalog", ""),
            arguments.get("schema", ""),
            arguments.get("table", ""),
        )
    if tool_name == "get_columns":
        meta = catalog.describe_table(
            arguments.get("catalog", ""),
            arguments.get("schema", ""),
            arguments.get("table", ""),
        )
        return meta if "error" in meta else {"columns": meta["columns"]}
    if tool_name == "search_tables":
        results = catalog.search_tables(arguments.get("query", ""))
        return {"query": arguments.get("query", ""), "count": len(results), "results": results}
    if tool_name == "get_table_relationships":
        from app.mcp.catalog import _parse_foreign_keys
        meta = catalog.describe_table(
            arguments.get("catalog", ""),
            arguments.get("schema", ""),
            arguments.get("table", ""),
        )
        if "error" in meta:
            return meta
        relationships = _parse_foreign_keys(meta["columns"])
        return {
            "catalog": meta["catalog"],
            "schema": meta["schema"],
            "table": meta["table"],
            "relationship_count": len(relationships),
            "relationships": relationships,
        }
    if tool_name == "validate_sql":
        return _validator.validate(arguments.get("sql", ""), SchemaContext()).model_dump()

    # Trino-backed tools (execution + profiling)
    if not settings.mcp_query_execution_enabled:
        return {"error": "Query execution is disabled on this server."}

    try:
        from app.services.trino import TrinoService
    except ImportError as exc:  # pragma: no cover
        return {"error": f"Trino service unavailable: {exc}"}

    trino = TrinoService()
    sql = arguments.get("sql", "")
    validation = _validator.validate(sql, SchemaContext())
    if not validation.is_valid:
        return {
            "error": "Query rejected before execution.",
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

    if tool_name == "execute_sql":
        limit = arguments.get("limit")
        cap = limit if limit is not None else settings.mcp_default_limit
        try:
            return trino.execute_query(sql, limit=cap)
        except Exception as exc:
            return {"error": f"Query execution failed: {exc}"}
    if tool_name == "get_table_sample":
        try:
            return trino.get_table_sample(
                arguments.get("catalog", ""),
                arguments.get("schema", ""),
                arguments.get("table", ""),
                limit=arguments.get("limit", 5),
            )
        except Exception as exc:
            return {"error": f"Could not sample table: {exc}"}
    if tool_name == "get_table_statistics":
        try:
            return trino.get_table_statistics(
                arguments.get("catalog", ""),
                arguments.get("schema", ""),
                arguments.get("table", ""),
            )
        except Exception as exc:
            return {"error": f"Could not get statistics: {exc}"}

    return {"error": f"Unknown MCP tool: {tool_name}"}


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Agent-facing client over MCP tools.

    Usage:
        mcp = MCPClient()
        mcp.list_catalogs()
        mcp.describe_table("hive", "sales", "orders")
        ctx = mcp.get_schema_context("Show me orders")
    """

    # Words that carry no table/schema signal when tokenizing a question.
    _FUNCTION_WORDS = frozenset({
        "a", "about", "all", "an", "and", "any", "are", "as", "at", "be",
        "by", "can", "could", "do", "does", "for", "from", "get", "give",
        "has", "have", "how", "in", "is", "it", "its", "list", "many",
        "me", "much", "of", "on", "or", "our", "please", "show", "some",
        "tell", "that", "the", "their", "then", "there", "they", "this",
        "to", "use", "we", "what", "which", "with", "would", "you", "your",
    })

    _FQN_RE = re.compile(r"([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)")
    _SCHEMA_TABLE_RE = re.compile(r"([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)")

    def __init__(self, transport: str | None = None) -> None:
        self._transport = (transport or settings.mcp_transport).lower()
        if self._transport not in {"sse", "inprocess"}:
            raise ValueError(
                f"Unknown MCP transport '{self._transport}' (expected 'sse' or 'inprocess')"
            )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        arguments = arguments or {}
        if self._transport == "sse":
            return _call_remote(tool_name, arguments)
        return _call_local(tool_name, arguments)

    # Metadata tools
    def list_catalogs(self) -> dict[str, Any]:
        return self._call_tool("list_catalogs")

    def list_schemas(self, catalog: str) -> dict[str, Any]:
        return self._call_tool("list_schemas", {"catalog": catalog})

    def list_tables(self, catalog: str, schema: str) -> dict[str, Any]:
        return self._call_tool("list_tables", {"catalog": catalog, "schema": schema})

    def describe_table(self, catalog: str, schema: str, table: str) -> dict[str, Any]:
        return self._call_tool(
            "describe_table",
            {"catalog": catalog, "schema": schema, "table": table},
        )

    def get_columns(self, catalog: str, schema: str, table: str) -> dict[str, Any]:
        return self._call_tool(
            "get_columns",
            {"catalog": catalog, "schema": schema, "table": table},
        )

    def search_tables(self, query: str) -> dict[str, Any]:
        return self._call_tool("search_tables", {"query": query})

    # Query tools
    def validate_sql(self, sql: str) -> dict[str, Any]:
        return self._call_tool("validate_sql", {"sql": sql})

    def execute_sql(self, sql: str, limit: int | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"sql": sql}
        if limit is not None:
            args["limit"] = limit
        return self._call_tool("execute_sql", args)

    # Profiling tools
    def get_table_sample(
        self, catalog: str, schema: str, table: str, limit: int = 5
    ) -> dict[str, Any]:
        return self._call_tool(
            "get_table_sample",
            {"catalog": catalog, "schema": schema, "table": table, "limit": limit},
        )

    def get_table_statistics(
        self, catalog: str, schema: str, table: str
    ) -> dict[str, Any]:
        return self._call_tool(
            "get_table_statistics",
            {"catalog": catalog, "schema": schema, "table": table},
        )

    def get_table_relationships(
        self, catalog: str, schema: str, table: str
    ) -> dict[str, Any]:
        return self._call_tool(
            "get_table_relationships",
            {"catalog": catalog, "schema": schema, "table": table},
        )

    # ------------------------------------------------------------------
    # Schema-context builder (used by the orchestrator)
    # ------------------------------------------------------------------

    def get_schema_context(self, question: str) -> SchemaContext:
        """
        Given a natural language question, return all relevant TableSchema
        objects to feed the SQL Agent.
        """
        try:
            candidate_tables = self._identify_candidate_tables(question)
            schemas: list[TableSchema] = []

            for cat, schema, table in candidate_tables:
                meta = self.describe_table(cat, schema, table)
                if "error" in meta:
                    logger.warning("table_not_found", fqn=f"{cat}.{schema}.{table}")
                    continue

                columns = [
                    ColumnInfo(
                        name=col["name"],
                        data_type=col["data_type"],
                        nullable=col.get("nullable", True),
                        comment=col.get("comment"),
                    )
                    for col in meta["columns"]
                ]

                schemas.append(
                    TableSchema(
                        catalog=cat,
                        schema=schema,
                        table_name=table,
                        columns=columns,
                        comment=meta.get("comment"),
                    )
                )

            logger.info(
                "schema_context_built",
                question=question[:60],
                tables_found=len(schemas),
            )
            return SchemaContext(tables=schemas)

        except Exception as exc:
            raise MCPError(f"Schema retrieval failed: {exc}") from exc

    def _identify_candidate_tables(
        self, question: str
    ) -> list[tuple[str, str, str]]:
        """
        Identify candidate (catalog, schema, table) tuples for a question.

        Strategy:
        1. Honor explicit references: ``catalog.schema.table`` and
           ``schema.table`` mentions are resolved exactly and returned alone.
        2. Otherwise tokenize the question and score every known table:
           exact table-name token (+3), substring (+2), schema-name token (+1).
           If a schema is explicitly named, non-matching schemas are pruned.
        3. Ties prefer the configured default schema (e.g. ``tiny``) so generic
           questions resolve to the demo dataset instead of the biggest copy.
        4. Fall back to core demo tables only when nothing matches.
        """
        all_tables = self._list_all_tables()
        lower = question.lower()

        # 1) Explicit fully-qualified reference: catalog.schema.table
        m = self._FQN_RE.search(lower)
        if m:
            catalog, schema, table = m.groups()
            hit = [
                (c, s, t) for (c, s, t) in all_tables
                if c.lower() == catalog and s.lower() == schema and t.lower() == table
            ]
            if hit:
                logger.info("mcp_explicit_fqn_match", fqn=f"{catalog}.{schema}.{table}")
                return hit[:5]

        # 2) Explicit schema.table reference
        m = self._SCHEMA_TABLE_RE.search(lower)
        if m:
            schema, table = m.groups()
            hit = [
                (c, s, t) for (c, s, t) in all_tables
                if s.lower() == schema and t.lower() == table
            ]
            if hit:
                logger.info("mcp_explicit_schema_table_match", schema=schema, table=table)
                return hit[:5]

        tokens = self._tokens(question)
        if not tokens:
            return self._default_tables()

        schema_names = {s.lower() for (_, s, _) in all_tables}
        mentioned_schema = next((tk for tk in tokens if tk in schema_names), None)

        scored: dict[str, tuple[tuple[str, str, str], float, bool]] = {}
        for catalog, schema, table in all_tables:
            tlower, slower = table.lower(), schema.lower()
            if mentioned_schema and slower != mentioned_schema:
                continue
            score = 0.0
            exact = False
            for tk in tokens:
                singular = tk[:-1] if tk.endswith("s") and len(tk) > 2 else None
                if tk == tlower or (singular and singular == tlower):
                    score += 3.0
                    exact = True
                elif tk in tlower or (len(tlower) > 2 and tlower in tk):
                    score += 2.0
                if tk == slower:
                    score += 1.0
            if score > 0:
                scored[f"{catalog}.{schema}.{table}"] = ((catalog, schema, table), score, exact)

        if not scored:
            return self._default_tables()

        default_schema = settings.trino_schema.lower()

        def rank(items: list[tuple[tuple[str, str, str], float, bool]]) -> list[tuple[str, str, str]]:
            def key(item: tuple[tuple[str, str, str], float, bool]) -> tuple:
                (catalog, schema, table), score, _exact = item
                schema_boost = 0 if schema.lower() == default_schema else 1
                return (schema_boost, -score, schema.lower(), table.lower())

            return [t for (t, _score, _exact) in sorted(items, key=key)][:5]

        if mentioned_schema is None:
            default_items = [
                item for item in scored.values() if item[0][1].lower() == default_schema
            ]
            if default_items:
                exact_items = [item for item in default_items if item[2]]
                return rank(exact_items or default_items)
        return rank(list(scored.values()))

    def _list_all_tables(self) -> list[tuple[str, str, str]]:
        """Enumerate (catalog, schema, table) tuples without column lookups."""
        out: list[tuple[str, str, str]] = []
        catalogs = self.list_catalogs().get("catalogs") or []
        for catalog in catalogs:
            schemas = self.list_schemas(catalog).get("schemas") or []
            for schema in schemas:
                if schema.lower() == "information_schema":
                    continue
                tables = self.list_tables(catalog, schema).get("tables") or []
                for table in tables:
                    out.append((catalog, schema, table))
                    if len(out) >= 400:
                        return out
        return out

    def _tokens(self, question: str) -> list[str]:
        """Split a question into meaningful lowercase tokens."""
        parts = re.split(r"[^a-z0-9_]+", question.lower())
        return [p for p in parts if len(p) > 1 and p not in self._FUNCTION_WORDS]

    @staticmethod
    def _default_tables() -> list[tuple[str, str, str]]:
        return [
            ("hive", "sales", "orders"),
            ("hive", "sales", "customers"),
            ("hive", "sales", "products"),
        ]