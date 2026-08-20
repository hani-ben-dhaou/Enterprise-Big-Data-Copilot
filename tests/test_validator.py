"""
Tests for the SQL Validator.

These tests run without Ollama or Qdrant — pure unit tests.
"""

import pytest

from app.core.models import SchemaContext, TableSchema, ColumnInfo, ValidationStatus
from app.validator.sql_validator import SQLValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema_context() -> SchemaContext:
    return SchemaContext(
        tables=[
            TableSchema(
                catalog="hive",
                schema="sales",
                table_name="orders",
                columns=[
                    ColumnInfo(name="order_id", data_type="BIGINT", nullable=False),
                    ColumnInfo(name="customer_id", data_type="BIGINT", nullable=False),
                    ColumnInfo(name="total_amount", data_type="DECIMAL(15,2)"),
                    ColumnInfo(name="order_date", data_type="DATE"),
                    ColumnInfo(name="status", data_type="VARCHAR"),
                ],
            )
        ]
    )


@pytest.fixture
def validator() -> SQLValidator:
    return SQLValidator()


# ---------------------------------------------------------------------------
# Valid queries
# ---------------------------------------------------------------------------

class TestValidQueries:
    def test_simple_select(self, validator, schema_context):
        sql = "SELECT order_id, total_amount FROM hive.sales.orders LIMIT 100"
        result = validator.validate(sql, schema_context)
        assert result.is_valid

    def test_select_with_where(self, validator, schema_context):
        sql = (
            "SELECT order_id, customer_id, total_amount "
            "FROM hive.sales.orders "
            "WHERE status = 'DELIVERED' AND order_date >= DATE '2024-01-01' "
            "LIMIT 500"
        )
        result = validator.validate(sql, schema_context)
        assert result.is_valid

    def test_select_with_aggregation(self, validator, schema_context):
        sql = (
            "SELECT customer_id, SUM(total_amount) AS total_revenue "
            "FROM hive.sales.orders "
            "GROUP BY customer_id "
            "ORDER BY total_revenue DESC "
            "LIMIT 10"
        )
        result = validator.validate(sql, schema_context)
        assert result.is_valid

    def test_cte_query(self, validator, schema_context):
        sql = """
        WITH filtered AS (
            SELECT customer_id, total_amount
            FROM hive.sales.orders
            WHERE status = 'DELIVERED'
        )
        SELECT customer_id, SUM(total_amount) AS revenue
        FROM filtered
        GROUP BY customer_id
        LIMIT 100
        """
        result = validator.validate(sql, schema_context)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Forbidden operations
# ---------------------------------------------------------------------------

class TestForbiddenOperations:
    def test_blocks_drop(self, validator, schema_context):
        sql = "DROP TABLE hive.sales.orders"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid
        assert result.status == ValidationStatus.INVALID

    def test_blocks_delete(self, validator, schema_context):
        sql = "DELETE FROM hive.sales.orders WHERE order_id = 1"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid

    def test_blocks_insert(self, validator, schema_context):
        sql = "INSERT INTO hive.sales.orders VALUES (1, 2, 100.0, '2024-01-01', 'NEW')"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid

    def test_blocks_truncate(self, validator, schema_context):
        sql = "TRUNCATE TABLE hive.sales.orders"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid

    def test_blocks_create(self, validator, schema_context):
        sql = "CREATE TABLE evil_table AS SELECT * FROM hive.sales.orders"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_missing_limit_warns(self, validator, schema_context):
        sql = "SELECT * FROM hive.sales.orders"
        result = validator.validate(sql, schema_context)
        # Should be valid but with warnings
        assert result.status in (ValidationStatus.VALID, ValidationStatus.WARNING)
        assert any("LIMIT" in w for w in result.warnings)

    def test_unknown_table_is_error(self, validator, schema_context):
        sql = "SELECT * FROM hive.sales.nonexistent_table LIMIT 10"
        result = validator.validate(sql, schema_context)
        assert result.status == ValidationStatus.INVALID
        assert any("not found" in e for e in result.errors)

    def test_mismatched_fqn_is_error(self, validator, schema_context):
        # Bare table name matches but the FQN points at the wrong catalog/schema.
        sql = "SELECT * FROM tpch.tiny.orders LIMIT 10"
        result = validator.validate(sql, schema_context)
        assert result.status == ValidationStatus.INVALID
        assert any("tpch.tiny.orders" in e for e in result.errors)

    def test_cte_not_flagged_as_unknown_table(self, validator, schema_context):
        sql = (
            "WITH recent AS (SELECT order_id FROM hive.sales.orders) "
            "SELECT order_id FROM recent LIMIT 10"
        )
        result = validator.validate(sql, schema_context)
        assert result.status == ValidationStatus.VALID or result.status == ValidationStatus.WARNING
        assert not any("recent" in e for e in result.errors)

    def test_unknown_column_is_error(self, validator, schema_context):
        sql = "SELECT order_id, totl FROM hive.sales.orders LIMIT 10"
        result = validator.validate(sql, schema_context)
        assert result.status == ValidationStatus.INVALID
        assert any("Column 'totl'" in e for e in result.errors)

    def test_qualified_unknown_column_is_error(self, validator, schema_context):
        sql = "SELECT orders.totl FROM hive.sales.orders LIMIT 10"
        result = validator.validate(sql, schema_context)
        assert result.status == ValidationStatus.INVALID
        assert any("totl" in e for e in result.errors)

    def test_select_alias_not_flagged_as_column(self, validator, schema_context):
        sql = (
            "SELECT customer_id, SUM(total_amount) AS revenue "
            "FROM hive.sales.orders GROUP BY customer_id "
            "ORDER BY revenue DESC LIMIT 10"
        )
        result = validator.validate(sql, schema_context)
        assert not any("revenue" in w for w in result.warnings)

    def test_forbidden_word_in_string_literal_not_blocked(self, validator, schema_context):
        sql = (
            "SELECT order_id FROM hive.sales.orders "
            "WHERE status = 'DROP TABLE important' LIMIT 5"
        )
        result = validator.validate(sql, schema_context)
        assert result.is_valid
        assert not any("Forbidden" in w for w in result.errors)

    def test_forbidden_word_in_comment_not_blocked(self, validator, schema_context):
        sql = (
            "SELECT order_id FROM hive.sales.orders -- legacy UPDATE note\n"
            "LIMIT 10"
        )
        result = validator.validate(sql, schema_context)
        assert result.is_valid
        assert not any("Forbidden" in w for w in result.errors)

    def test_real_delete_still_blocked(self, validator, schema_context):
        sql = "DELETE FROM hive.sales.orders WHERE order_id = 1"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_invalid_sql_fails(self, validator, schema_context):
        sql = "SELECT FROM WHERE"
        result = validator.validate(sql, schema_context)
        assert not result.is_valid

    def test_empty_sql_fails(self, validator, schema_context):
        sql = ""
        result = validator.validate(sql, schema_context)
        assert not result.is_valid