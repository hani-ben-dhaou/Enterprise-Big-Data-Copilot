"""
Shared domain models used across all modules.
These are the canonical data contracts for V1.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


class SQLDialect(str, Enum):
    TRINO = "trino"          # V1 only
    # HIVE = "hive"          # V2
    # SPARK = "spark"        # V2


# ---------------------------------------------------------------------------
# Schema / MCP Models
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    comment: str | None = None


class TableSchema(BaseModel):
    catalog: str
    schema_name: str = Field(alias="schema")
    table_name: str
    columns: list[ColumnInfo]
    comment: str | None = None
    row_count_estimate: int | None = None

    model_config = {"populate_by_name": True}


class SchemaContext(BaseModel):
    """Aggregated schema context passed to the SQL Agent."""
    tables: list[TableSchema] = Field(default_factory=list)
    raw_ddl: str | None = None


# ---------------------------------------------------------------------------
# RAG Models
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContext(BaseModel):
    """Aggregated RAG context passed to the SQL Agent."""
    chunks: list[DocumentChunk] = Field(default_factory=list)

    def as_text(self) -> str:
        if not self.chunks:
            return ""
        return "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in self.chunks
        )


# ---------------------------------------------------------------------------
# SQL Agent Models
# ---------------------------------------------------------------------------

class GeneratedSQL(BaseModel):
    sql: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    dialect: SQLDialect = SQLDialect.TRINO


# ---------------------------------------------------------------------------
# Validator Models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    status: ValidationStatus
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.INVALID


# ---------------------------------------------------------------------------
# Pipeline State (LangGraph node state)
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Mutable state object threaded through the LangGraph pipeline."""

    # Input
    question: str

    # Intermediate
    rag_context: RAGContext | None = None
    schema_context: SchemaContext | None = None
    generated_sql: GeneratedSQL | None = None
    validation_result: ValidationResult | None = None
    regeneration_count: int = 0
    # Best-effort query execution result (columns + rows), when enabled
    execution_result: dict[str, Any] | None = None

    # Output
    final_response: "CopilotResponse | None" = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Final API Response
# ---------------------------------------------------------------------------

class CopilotResponse(BaseModel):
    """The canonical API response returned to callers."""
    question: str
    sql: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    dialect: SQLDialect = SQLDialect.TRINO

    # Query execution (best-effort, only present when the platform ran it)
    results: list[list[Any]] | None = None
    execution: dict[str, Any] | None = None