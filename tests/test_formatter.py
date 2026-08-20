"""Tests for the in-memory catalog (MCP layer)."""

import pytest

from app.mcp.catalog import InMemoryCatalog


@pytest.fixture
def catalog() -> InMemoryCatalog:
    return InMemoryCatalog()


class TestInMemoryCatalog:
    def test_list_catalogs(self, catalog):
        catalogs = catalog.list_catalogs()
        assert "hive" in catalogs

    def test_list_schemas(self, catalog):
        schemas = catalog.list_schemas("hive")
        assert "sales" in schemas
        assert "analytics" in schemas

    def test_list_tables(self, catalog):
        tables = catalog.list_tables("hive", "sales")
        assert "orders" in tables
        assert "customers" in tables

    def test_describe_table_returns_columns(self, catalog):
        meta = catalog.describe_table("hive", "sales", "orders")
        assert meta["table"] == "orders"
        assert len(meta["columns"]) > 0
        col_names = [c["name"] for c in meta["columns"]]
        assert "order_id" in col_names
        assert "total_amount" in col_names

    def test_describe_missing_table(self, catalog):
        meta = catalog.describe_table("hive", "sales", "does_not_exist")
        assert "error" in meta

    def test_search_tables_by_table_name(self, catalog):
        results = catalog.search_tables("orders")
        assert any(r["table"] == "orders" for r in results)

    def test_search_tables_by_column_name(self, catalog):
        results = catalog.search_tables("customer_id")
        assert len(results) > 0

    def test_search_tables_no_match(self, catalog):
        results = catalog.search_tables("xyz_nonexistent_zzz")
        assert results == []