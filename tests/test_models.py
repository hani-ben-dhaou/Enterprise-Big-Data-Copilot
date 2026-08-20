"""Tests for core Pydantic domain models."""

import pytest
from pydantic import ValidationError

from app.core.models import (
    ColumnInfo,
    CopilotResponse,
    DocumentChunk,
    GeneratedSQL,
    PipelineState,
    RAGContext,
    SchemaContext,
    SQLDialect,
    TableSchema,
    ValidationResult,
    ValidationStatus,
)


class TestColumnInfo:
    def test_basic(self):
        col = ColumnInfo(name="order_id", data_type="BIGINT", nullable=False)
        assert col.name == "order_id"
        assert col.nullable is False

    def test_default_nullable(self):
        col = ColumnInfo(name="status", data_type="VARCHAR")
        assert col.nullable is True


class TestTableSchema:
    def test_alias_schema(self):
        t = TableSchema(
            catalog="hive",
            schema="sales",
            table_name="orders",
            columns=[ColumnInfo(name="id", data_type="BIGINT")],
        )
        assert t.schema_name == "sales"

    def test_populate_by_name(self):
        t = TableSchema(
            catalog="hive",
            schema_name="sales",
            table_name="orders",
            columns=[],
        )
        assert t.schema_name == "sales"


class TestRAGContext:
    def test_as_text_empty(self):
        ctx = RAGContext()
        assert ctx.as_text() == ""

    def test_as_text_with_chunks(self):
        ctx = RAGContext(chunks=[
            DocumentChunk(content="Use DATE_TRUNC.", source="trino.md", score=0.9),
            DocumentChunk(content="Partition pruning is important.", source="hive.md", score=0.8),
        ])
        text = ctx.as_text()
        assert "DATE_TRUNC" in text
        assert "trino.md" in text


class TestGeneratedSQL:
    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            GeneratedSQL(sql="SELECT 1", explanation="x", confidence=1.5)

    def test_valid(self):
        g = GeneratedSQL(sql="SELECT 1", explanation="Test", confidence=0.85)
        assert g.dialect == SQLDialect.TRINO


class TestValidationResult:
    def test_is_valid(self):
        r = ValidationResult(status=ValidationStatus.VALID)
        assert r.is_valid is True

    def test_is_invalid(self):
        r = ValidationResult(status=ValidationStatus.INVALID, errors=["forbidden keyword"])
        assert r.is_valid is False

    def test_warning_is_valid(self):
        r = ValidationResult(status=ValidationStatus.WARNING, warnings=["no LIMIT"])
        assert r.is_valid is True


class TestPipelineState:
    def test_initial_state(self):
        s = PipelineState(question="How many orders?")
        assert s.regeneration_count == 0
        assert s.error is None
        assert s.final_response is None


class TestCopilotResponse:
    def test_full_response(self):
        r = CopilotResponse(
            question="Test?",
            sql="SELECT 1",
            explanation="Just a test.",
            confidence=0.99,
            warnings=[],
        )
        assert r.dialect == SQLDialect.TRINO
        d = r.model_dump()
        assert "sql" in d
        assert "warnings" in d