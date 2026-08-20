"""Tests for the MCPClient schema context builder."""

import pytest

from app.mcp.client import MCPClient
from app.core.models import SchemaContext


@pytest.fixture
def client() -> MCPClient:
    return MCPClient()


class TestMCPClient:
    def test_returns_schema_context(self, client):
        ctx = client.get_schema_context("show me orders")
        assert isinstance(ctx, SchemaContext)
        assert len(ctx.tables) > 0

    def test_finds_orders_table(self, client):
        ctx = client.get_schema_context("top customers by revenue")
        table_names = [t.table_name for t in ctx.tables]
        assert "orders" in table_names or "customers" in table_names

    def test_fallback_when_no_keyword_match(self, client):
        ctx = client.get_schema_context("xyzzy gibberish nonsense query")
        # fallback returns default core tables
        assert len(ctx.tables) > 0

    def test_table_has_columns(self, client):
        ctx = client.get_schema_context("show orders")
        for table in ctx.tables:
            assert len(table.columns) > 0

    def test_caps_at_five_tables(self, client):
        # Even a broad keyword shouldn't return more than 5 tables
        ctx = client.get_schema_context("id date amount revenue event order")
        assert len(ctx.tables) <= 5


class _TpchCatalog:
    """Fake live metadata backend with the same table in several schemas."""

    def list_catalogs(self):
        return ["tpch"]

    def list_schemas(self, catalog):
        if catalog != "tpch":
            return []
        return ["tiny", "sf1", "sf100", "information_schema"]

    def list_tables(self, catalog, schema):
        if schema in ("tiny", "sf1", "sf100"):
            return ["orders", "customer", "lineitem", "partsupp", "part"]
        return []

    def describe_table(self, catalog, schema, table):
        return {
            "catalog": catalog,
            "schema": schema,
            "table": table,
            "comment": "",
            "columns": [
                {"name": f"{table}_key", "data_type": "BIGINT", "nullable": False}
            ],
            "fully_qualified_name": f"{catalog}.{schema}.{table}",
        }

    def search_tables(self, keyword):
        return []


class TestSchemaAwareCandidateSelection:
    @pytest.fixture
    def tpch_client(self, monkeypatch):
        monkeypatch.setattr("app.mcp.client._local_catalog", _TpchCatalog())
        return MCPClient(transport="inprocess")

    def test_explicit_fqn_picks_exact_table(self, tpch_client):
        ctx = tpch_client.get_schema_context(
            "How many orders are there in the tpch.tiny.orders table?"
        )
        tables = [(t.catalog, t.schema_name, t.table_name) for t in ctx.tables]
        assert tables == [("tpch", "tiny", "orders")]

    def test_explicit_fqn_not_the_biggest_schema(self, tpch_client):
        ctx = tpch_client.get_schema_context("count tpch.tiny.orders")
        tables = [(t.schema_name, t.table_name) for t in ctx.tables]
        assert all(schema == "tiny" for schema, _ in tables)

    def test_schema_table_two_part_reference(self, tpch_client):
        ctx = tpch_client.get_schema_context("orders in tiny.orders")
        tables = [(t.schema_name, t.table_name) for t in ctx.tables]
        assert tables == [("tiny", "orders")]

    def test_named_schema_filters_other_schemas(self, tpch_client):
        ctx = tpch_client.get_schema_context("how many orders are in the tiny schema")
        tables = [(t.catalog, t.schema_name, t.table_name) for t in ctx.tables]
        assert tables
        assert all(schema == "tiny" for _, schema, _ in tables)
        assert ("tpch", "tiny", "orders") in tables

    def test_generic_prefers_default_schema(self, tpch_client):
        ctx = tpch_client.get_schema_context("how many orders are there?")
        tables = [(t.catalog, t.schema_name, t.table_name) for t in ctx.tables]
        assert tables == [("tpch", "tiny", "orders")]

    def test_generic_never_spans_scale_schemas(self, tpch_client):
        ctx = tpch_client.get_schema_context("what is the total quantity of partsupp?")
        tables = [(t.schema_name, t.table_name) for t in ctx.tables]
        assert tables
        assert all(schema == "tiny" for schema, _ in tables)
        assert ("tiny", "partsupp") in tables

    def test_no_match_falls_back_to_default_tables(self, tpch_client):
        ctx = tpch_client.get_schema_context("xyzzy gibberish nonsense")
        tables = [(t.catalog, t.schema_name, t.table_name) for t in ctx.tables]
        assert len(ctx.tables) > 0
        assert ("hive", "sales", "orders") in tables

    def test_plural_table_name_matches_singular(self, tpch_client):
        ctx = tpch_client.get_schema_context("how many customers from nation 5?")
        tables = [(t.schema_name, t.table_name) for t in ctx.tables]
        assert tables
        assert tables[0] == ("tiny", "customer")