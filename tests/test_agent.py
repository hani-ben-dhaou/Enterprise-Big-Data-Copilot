"""
Unit tests for the SQL Agent's LLM-output parsing layer.

Only pure helpers / static methods are tested here — no Ollama calls.
"""

import pytest

from app.agent.sql_agent import SQLAgent, _build_schema_text, _extract_json
from app.core.exceptions import SQLGenerationError
from app.core.models import ColumnInfo, GeneratedSQL, RAGContext, SchemaContext, TableSchema

ORDER_TABLE = TableSchema(
    catalog="hive",
    schema_name="sales",
    table_name="orders",
    columns=[
        ColumnInfo(name="order_id", data_type="bigint"),
        ColumnInfo(name="total_amount", data_type="double"),
        ColumnInfo(name="status", data_type="varchar"),
    ],
)


def make_schema(*tables: TableSchema) -> SchemaContext:
    return SchemaContext(tables=list(tables))


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_from_fenced_block():
    raw = '```json\n{"sql": "SELECT 1", "explanation": "ok", "confidence": 0.9}\n```'
    parsed = _extract_json(raw)
    assert parsed["sql"] == "SELECT 1"
    assert parsed["confidence"] == 0.9


def test_extract_json_plain():
    raw = '{"sql": "SELECT 1", "explanation": "ok"}'
    parsed = _extract_json(raw)
    assert parsed["sql"] == "SELECT 1"


def test_extract_json_raises_without_braces():
    with pytest.raises(SQLGenerationError):
        _extract_json("the model rambled on with no json at all")


def test_extract_json_survives_trailing_text():
    raw = '{"sql": "SELECT 1"} and some trailing chatter'
    parsed = _extract_json(raw)
    assert parsed["sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

def test_parse_response_ok():
    result = SQLAgent._parse_response(
        {"sql": "SELECT 1", "explanation": "just one", "confidence": 0.85}
    )
    assert isinstance(result, GeneratedSQL)
    assert result.sql == "SELECT 1"
    assert result.confidence == 0.85


def test_parse_response_tolerates_non_numeric_confidence():
    result = SQLAgent._parse_response(
        {"sql": "SELECT 1", "confidence": "high"}
    )
    assert result.confidence == 0.7


def test_parse_response_clamps_out_of_range_confidence():
    result = SQLAgent._parse_response(
        {"sql": "SELECT 1", "confidence": 5}
    )
    assert result.confidence == 1.0


def test_parse_response_defaults_missing_fields():
    result = SQLAgent._parse_response({"sql": "SELECT 1"})
    assert result.explanation
    assert result.confidence == 0.7


def test_parse_response_rejects_empty_sql():
    with pytest.raises(SQLGenerationError):
        SQLAgent._parse_response({"sql": "   "})


# ---------------------------------------------------------------------------
# _build_schema_text
# ---------------------------------------------------------------------------

def test_build_schema_text_empty():
    assert _build_schema_text(SchemaContext()) == "No schema available."


def test_build_schema_text_lists_fqn_and_columns():
    text = _build_schema_text(make_schema(ORDER_TABLE))
    assert "hive.sales.orders" in text
    assert "order_id" in text
    assert "total_amount" in text