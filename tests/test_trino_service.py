"""Tests for the TrinoService (Trino client mocked — no live Trino needed)."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.trino import TrinoService, quote_identifier


@pytest.fixture
def service() -> TrinoService:
    return TrinoService(
        host="localhost", port=8080, user="test",
        catalog="hive", schema="sales",
    )


def _mock_connection(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


class TestIdentifierQuoting:
    def test_plain_identifier(self):
        assert quote_identifier("sales") == '"sales"'

    def test_escapes_double_quote(self):
        assert quote_identifier('weird"name') == '"weird""name"'

    def test_empty_string(self):
        assert quote_identifier("") == '""'


class TestExecuteQuery:
    def test_execute_with_limit(self, service):
        cursor = MagicMock()
        cursor.description = [("id",), ("name",)]
        cursor.fetchmany.return_value = [[1, "a"], [2, "b"]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            result = service.execute_query("SELECT 1", limit=10)

        assert result["columns"] == ["id", "name"]
        assert result["rows"] == [[1, "a"], [2, "b"]]
        assert result["row_count"] == 2
        assert result["truncated"] is False
        cursor.fetchmany.assert_called_once_with(10)
        conn.close.assert_called_once()

    def test_execute_without_limit_fetches_all(self, service):
        cursor = MagicMock()
        cursor.description = [("id",)]
        cursor.fetchall.return_value = [[1], [2], [3]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            result = service.execute_query("SELECT 1", limit=None)

        assert result["row_count"] == 3
        cursor.fetchall.assert_called_once()
        cursor.fetchmany.assert_not_called()

    def test_limit_respected_and_hard_capped(self, service):
        cursor = MagicMock()
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            # limit above hard_cap is clamped to 10_000
            service.execute_query("SELECT 1", limit=999_999)
            assert cursor.fetchmany.call_args[0][0] == 10_000

    def test_truncated_flag(self, service):
        cursor = MagicMock()
        cursor.description = [("id",)]
        cursor.fetchmany.return_value = [[1], [2]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            result = service.execute_query("SELECT 1", limit=2)

        assert result["truncated"] is True


class TestIntrospection:
    def test_list_catalogs(self, service):
        cursor = MagicMock()
        cursor.fetchall.return_value = [["hive"], ["tpch"]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            assert service.list_catalogs() == ["hive", "tpch"]

    def test_list_schemas_quotes_catalog(self, service):
        cursor = MagicMock()
        cursor.fetchall.return_value = [["sales"], ["analytics"]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            schemas = service.list_schemas("hive")

        assert schemas == ["sales", "analytics"]
        sql = cursor.execute.call_args[0][0]
        assert '"hive".information_schema.schemata' in sql

    def test_list_tables_passes_schema_param(self, service):
        cursor = MagicMock()
        cursor.fetchall.return_value = [["orders"]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            tables = service.list_tables("hive", "sales")

        assert tables == ["orders"]
        args, kwargs = cursor.execute.call_args
        assert args[0].startswith('SELECT table_name FROM "hive".information_schema.tables')
        assert args[1] == ["sales"]

    def test_get_columns_maps_nullability(self, service):
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ["order_id", "bigint", "NO", None],
            ["status", "varchar", "YES", "Order status"],
        ]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            columns = service.get_columns("hive", "sales", "orders")

        assert columns[0]["nullable"] is False
        assert columns[1]["nullable"] is True
        assert columns[1]["comment"] == "Order status"

    def test_get_table_sample_uses_quoted_fqn(self, service):
        cursor = MagicMock()
        cursor.description = [("id",)]
        cursor.fetchmany.return_value = [[1]]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            service.get_table_sample("hive", "sales", "orders", limit=5)

        sql = cursor.execute.call_args[0][0]
        assert 'FROM "hive"."sales"."orders"' in sql
        assert "LIMIT 5" in sql


class TestStatistics:
    def test_show_stats_row_count_estimate(self, service):
        """SHOW STATS summary row: (column_name, data_size,
        distinct_values_count, nulls_fraction, row_count, low_value,
        high_value) — row_count is at index 4, not 5."""
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ["total_amount", 123456, 500, 0.1, 0, None, None],
            ["status", 456, 4, 0.2, 0, None, "DELIVERED"],
            [None, None, None, 0.0, 98765, None, None],  # summary row
        ]
        conn = _mock_connection(cursor)

        with patch.object(TrinoService, "_connection", return_value=conn):
            result = service.get_table_statistics("hive", "sales", "orders")

        assert result["estimate_available"] is True
        assert result["row_count_estimate"] == 98765

    def test_show_stats_unavailable_when_query_raises(self, service):
        with patch.object(
            TrinoService, "_query", side_effect=RuntimeError("catalog down")
        ):
            result = service.get_table_statistics("hive", "sales", "orders")

        assert result["estimate_available"] is False
        assert "error" in result