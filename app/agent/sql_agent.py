"""
SQL Agent — generates Trino SQL using Ollama + structured prompts.

Responsibilities:
  - Build prompt from schema + RAG context
  - Call Ollama LLM
  - Parse structured JSON response
  - Return GeneratedSQL

This module has NO knowledge of the orchestrator pipeline — it
receives context objects and returns a GeneratedSQL.
"""

from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.exceptions import SQLGenerationError
from app.core.logging import get_logger
from app.core.models import (
    GeneratedSQL,
    RAGContext,
    SchemaContext,
    SQLDialect,
    TableSchema,
)
from app.agent.prompts import (
    SQL_SYSTEM_PROMPT,
    SQL_USER_TEMPLATE,
    REGENERATION_SYSTEM_PROMPT,
    REGENERATION_USER_TEMPLATE,
)

logger = get_logger(__name__)
settings = get_settings()


def _build_schema_text(schema_context: SchemaContext) -> str:
    """Convert SchemaContext into a compact human-readable schema block."""
    if not schema_context.tables:
        return "No schema available."

    lines: list[str] = []
    for table in schema_context.tables:
        fqn = f"{table.catalog}.{table.schema_name}.{table.table_name}"
        lines.append(f"\nTable: {fqn}")
        if table.comment:
            lines.append(f"  Description: {table.comment}")
        lines.append("  Columns:")
        for col in table.columns:
            nullable_flag = "" if col.nullable else " NOT NULL"
            comment = f"  -- {col.comment}" if col.comment else ""
            lines.append(f"    {col.name}  {col.data_type}{nullable_flag}{comment}")

    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """
    Extract JSON from LLM output.
    Handles cases where the model wraps output in markdown code fences.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find first { ... } block
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise SQLGenerationError(f"No JSON found in LLM output: {raw[:200]}")

    return json.loads(clean[start:end])


class SQLAgent:
    """
    Stateless SQL generation agent.

    Instantiate once and call generate() / regenerate() per request.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,        # Low temp → more deterministic SQL
            format="json",          # Ollama structured output mode
            timeout=settings.llm_timeout,  # prevent a hung Ollama stalling the API
        )

    def generate(
        self,
        question: str,
        schema_context: SchemaContext,
        rag_context: RAGContext,
    ) -> GeneratedSQL:
        """Generate SQL from scratch."""
        schema_text = _build_schema_text(schema_context)
        rag_text = rag_context.as_text() or "No documentation context available."

        user_content = SQL_USER_TEMPLATE.format(
            question=question,
            schema_context=schema_text,
            rag_context=rag_text,
        )

        messages = [
            SystemMessage(content=SQL_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        logger.info("sql_agent_generating", question=question[:80])

        try:
            response = self._llm.invoke(messages)
            raw = response.content
            parsed = _extract_json(raw)
        except Exception as exc:
            raise SQLGenerationError(f"LLM call failed: {exc}") from exc

        return self._parse_response(parsed)

    def regenerate(
        self,
        question: str,
        failed_sql: str,
        errors: list[str],
        schema_context: SchemaContext,
    ) -> GeneratedSQL:
        """Regenerate SQL after a validation failure."""
        schema_text = _build_schema_text(schema_context)

        user_content = REGENERATION_USER_TEMPLATE.format(
            question=question,
            failed_sql=failed_sql,
            errors="\n".join(f"- {e}" for e in errors),
            schema_context=schema_text,
        )

        messages = [
            SystemMessage(content=REGENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        logger.info("sql_agent_regenerating", errors=errors)

        try:
            response = self._llm.invoke(messages)
            raw = response.content
            parsed = _extract_json(raw)
        except Exception as exc:
            raise SQLGenerationError(f"Regeneration LLM call failed: {exc}") from exc

        return self._parse_response(parsed)

    @staticmethod
    def _parse_response(parsed: dict) -> GeneratedSQL:
        sql = parsed.get("sql", "").strip()
        explanation = parsed.get("explanation", "No explanation provided.").strip()

        # The LLM is not trusted to return a parseable float; a bad value
        # must degrade to the default instead of crashing the pipeline.
        try:
            confidence = float(parsed.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7

        if not sql:
            raise SQLGenerationError("LLM returned empty SQL.")

        return GeneratedSQL(
            sql=sql,
            explanation=explanation,
            confidence=max(0.0, min(1.0, confidence)),
            dialect=SQLDialect.TRINO,
        )