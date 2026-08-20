"""Tests for the MCP catalog services (in-memory + Trino-backed)."""

from unittest.mock import MagicMock, patch

import pytest

from app.mcp.catalog import InMemoryCatalog, TrinoCatalog, get_catalog_service


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

    def test_list_schemas_unknown_catalog(self, catalog):
        assert catalog.list_schemas("nope") == []

    def test_list_tables(self, catalog):
        tables = catalog.list_tables("hive", "sales")
        assert "orders" in tables
        assert "customers" in tables

    def test_list_tables_unknown_schema(self, catalog):
        assert catalog.list_tables("hive", "nope") == []

    def test_describe_table_returns_columns(self, catalog):
        meta = catalog.describe_table("hive", "sales", "orders")
        assert meta["table"] == "orders"
        assert meta["fully_qualified_name"] == "hive.sales.orders"
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
        assert all("matched_columns" in r for r in results)

    def test_search_tables_no_match(self, catalog):
        results = catalog.search_tables("xyz_nonexistent_zzz")
        assert results == []


class TestTrinoCatalog:
    @pytest.fixture
    def service(self) -> MagicMock:
        svc = MagicMock()
        svc.list_catalogs.return_value = ["tpch", "system"]
        svc.list_schemas.return_value = ["tiny"]
        svc.list_tables.return_value = ["orders", "customers"]
        svc.get_columns.return_value = [
            {"name": "orderkey", "data_type": "bigint", "nullable": False, "comment": None},
            {"name": "custkey", "data_type": "bigint", "nullable": True, "comment": None},
        ]
        return svc

    def test_list_catalogs(self, service):
        assert TrinoCatalog(service).list_catalogs() == ["tpch", "system"]

    def test_list_schemas_and_tables(self, service):
        tc = TrinoCatalog(service)
        assert tc.list_schemas("tpch") == ["tiny"]
        assert tc.list_tables("tpch", "tiny") == ["orders", "customers"]

    def test_describe_table_maps_trino_columns(self, service):
        meta = TrinoCatalog(service).describe_table("tpch", "tiny", "orders")
        assert meta["table"] == "orders"
        assert meta["columns"][0]["name"] == "orderkey"
        assert meta["columns"][0]["nullable"] is False

    def test_describe_table_empty_columns_is_error(self, service):
        service.get_columns.return_value = []
        meta = TrinoCatalog(service).describe_table("tpch", "tiny", "nope")
        assert "error" in meta

    def test_describe_table_service_error_is_graceful(self, service):
        service.get_columns.side_effect = RuntimeError("connection refused")
        meta = TrinoCatalog(service).describe_table("tpch", "tiny", "orders")
        assert "error" in meta

    def test_search_tables(self, service):
        results = TrinoCatalog(service).search_tables("order")
        assert any(r["table"] == "orders" for r in results)


class TestGetCatalogService:
    @patch("app.mcp.catalog.settings")
    def test_inmemory_by_default(self, mock_settings):
        mock_settings.mcp_metadata_source = "inmemory"
        service = get_catalog_service()
        assert isinstance(service, InMemoryCatalog)

    @patch("app.mcp.catalog.settings")
    def test_trino_source_returns_trino_catalog(self, mock_settings):
        mock_settings.mcp_metadata_source = "trino"
        with patch("app.mcp.catalog.TrinoCatalog") as mock_cls:
            mock_cls.return_value = MagicMock()
            service = get_catalog_service()
            mock_cls.assert_called_once()
            assert service is mock_cls.return_value

    @patch("app.mcp.catalog.settings")
    def test_trino_unavailable_falls_back_to_inmemory(self, mock_settings):
        mock_settings.mcp_metadata_source = "trino"
        with patch("app.mcp.catalog.TrinoCatalog", side_effect=RuntimeError("down")):
            service = get_catalog_service()
        assert isinstance(service, InMemoryCatalog)