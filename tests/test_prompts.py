"""Tests for the SQL Agent prompt templates."""

from app.agent.prompts import (  # direct import — no LLM deps needed
    SQL_SYSTEM_PROMPT,
    SQL_USER_TEMPLATE,
    REGENERATION_SYSTEM_PROMPT,
    REGENERATION_USER_TEMPLATE,
)
from app.core.models import RAGContext, SchemaContext


class TestPromptTemplates:
    def test_system_prompt_contains_rules(self):
        assert "SELECT" in SQL_SYSTEM_PROMPT
        assert "fully-qualified" in SQL_SYSTEM_PROMPT
        assert "LIMIT" in SQL_SYSTEM_PROMPT
        assert "JSON" in SQL_SYSTEM_PROMPT

    def test_user_template_renders(self):
        rendered = SQL_USER_TEMPLATE.format(
            question="How many orders?",
            schema_context="Table: hive.sales.orders\n  Columns:\n    order_id BIGINT",
            rag_context="Use DATE_TRUNC for month truncation.",
        )
        assert "How many orders?" in rendered
        assert "hive.sales.orders" in rendered
        assert "DATE_TRUNC" in rendered

    def test_regeneration_template_renders(self):
        rendered = REGENERATION_USER_TEMPLATE.format(
            question="How many orders?",
            failed_sql="SELECT * FROM orders",
            errors="- Table 'orders' not fully qualified",
            schema_context="Table: hive.sales.orders",
        )
        assert "failed_sql" not in rendered  # placeholder replaced
        assert "SELECT * FROM orders" in rendered
        assert "not fully qualified" in rendered

    def test_regeneration_system_prompt(self):
        assert "validation errors" in REGENERATION_SYSTEM_PROMPT.lower()
        assert "JSON" in REGENERATION_SYSTEM_PROMPT