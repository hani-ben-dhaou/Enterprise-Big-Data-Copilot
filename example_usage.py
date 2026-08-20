"""
example_usage.py — demonstrates the copilot without a running Ollama instance.

This script shows how each module works independently and how they compose.
It mocks the LLM call so you can run it offline.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

# Ensure console rendering of unicode glyphs (✓ ✗ ─) works on Windows (cp1252).
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# 1. Schema Context — what the MCP layer returns
# ---------------------------------------------------------------------------

from app.mcp.client import MCPClient

mcp = MCPClient()
schema = mcp.get_schema_context("top customers by revenue last month")

print("=" * 60)
print("STEP 1 — MCP Schema Context")
print("=" * 60)
for table in schema.tables:
    fqn = f"{table.catalog}.{table.schema_name}.{table.table_name}"
    print(f"\n  Table: {fqn}")
    for col in table.columns[:4]:  # first 4 cols
        print(f"    {col.name:25} {col.data_type}")


# ---------------------------------------------------------------------------
# 2. Validator — standalone check
# ---------------------------------------------------------------------------

from app.core.models import SchemaContext
from app.validator.sql_validator import SQLValidator

validator = SQLValidator()

test_queries = [
    ("VALID SELECT", "SELECT customer_id, SUM(total_amount) AS revenue FROM hive.sales.orders GROUP BY customer_id ORDER BY revenue DESC LIMIT 10"),
    ("MISSING LIMIT", "SELECT * FROM hive.sales.orders"),
    ("FORBIDDEN DROP", "DROP TABLE hive.sales.orders"),
    ("BAD SQL", "SELECT FROM WHERE !!!"),
]

print("\n" + "=" * 60)
print("STEP 2 — SQL Validator")
print("=" * 60)

for label, sql in test_queries:
    result = validator.validate(sql, schema)
    icon = "✓" if result.is_valid else "✗"
    print(f"\n  [{icon}] {label}")
    print(f"      Status  : {result.status}")
    if result.errors:
        print(f"      Errors  : {result.errors}")
    if result.warnings:
        print(f"      Warnings: {result.warnings}")


# ---------------------------------------------------------------------------
# 3. Full pipeline with mocked LLM
# ---------------------------------------------------------------------------

from app.core.models import CopilotResponse

MOCK_LLM_RESPONSE = json.dumps({
    "sql": (
        "WITH monthly_orders AS (\n"
        "    SELECT\n"
        "        o.customer_id,\n"
        "        c.name         AS customer_name,\n"
        "        SUM(o.total_amount) AS total_revenue\n"
        "    FROM hive.sales.orders o\n"
        "    JOIN hive.sales.customers c ON o.customer_id = c.customer_id\n"
        "    WHERE o.status = 'DELIVERED'\n"
        "      AND o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1' MONTH)\n"
        "      AND o.order_date <  DATE_TRUNC('month', CURRENT_DATE)\n"
        "    GROUP BY o.customer_id, c.name\n"
        ")\n"
        "SELECT customer_id, customer_name, total_revenue\n"
        "FROM monthly_orders\n"
        "ORDER BY total_revenue DESC\n"
        "LIMIT 10"
    ),
    "explanation": (
        "Joins orders with customers to get names, filters to last calendar month "
        "using DATE_TRUNC, aggregates total revenue per customer, and returns the "
        "top 10 by descending revenue."
    ),
    "confidence": 0.93
})

print("\n" + "=" * 60)
print("STEP 3 — Full Pipeline (Ollama mocked)")
print("=" * 60)

mock_message = MagicMock()
mock_message.content = MOCK_LLM_RESPONSE

with patch("app.agent.sql_agent.ChatOllama") as MockLLM:
    instance = MockLLM.return_value
    instance.invoke.return_value = mock_message

    # Also mock RAG (no Qdrant needed)
    with patch("app.rag.retriever.RAGRetriever.retrieve") as mock_rag:
        from app.core.models import RAGContext, DocumentChunk
        mock_rag.return_value = RAGContext(chunks=[
            DocumentChunk(
                content="Use DATE_TRUNC to truncate timestamps to month boundaries.",
                source="docs/trino_reference.md",
                score=0.91,
            )
        ])

        from app.orchestrator.orchestrator import Orchestrator
        orchestrator = Orchestrator()
        response: CopilotResponse = orchestrator.run(
            "Show me the top 10 customers by revenue last month"
        )

print(f"\n  Question   : {response.question}")
print(f"  Confidence : {response.confidence}")
print(f"  Warnings   : {response.warnings}")
print(f"\n  SQL:\n")
for line in response.sql.split("\n"):
    print(f"    {line}")
print(f"\n  Explanation: {response.explanation}")

print("\n" + "=" * 60)
print("JSON Response (as returned by the API):")
print("=" * 60)
print(json.dumps(response.model_dump(), indent=2))