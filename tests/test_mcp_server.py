"""Tests for the MCP server tool functions (invoked directly, no transport)."""

from app.mcp import server as srv
from app.mcp.catalog import _parse_foreign_keys


class TestMetadataTools:
    def test_list_catalogs(self):
        result = srv.list_catalogs()
        assert "hive" in result["catalogs"]

    def test_list_schemas(self):
        result = srv.list_schemas("hive")
        assert "sales" in result["schemas"]

    def test_list_tables(self):
        result = srv.list_tables("hive", "sales")
        assert "orders" in result["tables"]
        assert "customers" in result["tables"]

    def test_describe_table(self):
        result = srv.describe_table("hive", "sales", "orders")
        assert result["table"] == "orders"
        assert len(result["columns"]) > 0
        assert "fully_qualified_name" in result

    def test_describe_missing_table(self):
        result = srv.describe_table("hive", "sales", "does_not_exist")
        assert "error" in result

    def test_get_columns(self):
        result = srv.get_columns("hive", "sales", "orders")
        names = [c["name"] for c in result["columns"]]
        assert "order_id" in names
        assert "total_amount" in names

    def test_search_tables(self):
        result = srv.search_tables("customer_id")
        assert result["count"] > 0
        assert any(r["table"] == "orders" for r in result["results"])


class TestQueryTools:
    def test_validate_sql_valid(self):
        result = srv.validate_sql(
            "SELECT * FROM hive.sales.orders LIMIT 10"
        )
        assert result["status"] in ("valid", "warning")

    def test_validate_sql_rejects_drop(self):
        result = srv.validate_sql("DROP TABLE hive.sales.orders")
        assert result["status"] == "invalid"

    def test_execute_sql_rejects_non_select(self):
        result = srv.execute_sql("DELETE FROM hive.sales.orders WHERE order_id = 1")
        assert "error" in result
        assert result["errors"]

    def test_execute_sql_invalid_syntax(self):
        result = srv.execute_sql("SELECT FROM WHERE !!!")
        assert "error" in result


class TestProfilingTools:
    def test_table_sample_offline_returns_graceful_error(self):
        # in-memory catalog has no row data → graceful message, no crash
        result = srv.get_table_sample("hive", "sales", "orders")
        assert "error" in result

    def test_table_statistics_schema_only(self):
        result = srv.get_table_statistics("hive", "sales", "orders")
        assert result["column_count"] == 8
        assert result["row_count_estimate"] is None
        assert result["estimate_available"] is False

    def test_table_statistics_missing_table(self):
        result = srv.get_table_statistics("hive", "sales", "nope")
        assert "error" in result

    def test_table_relationships(self):
        result = srv.get_table_relationships("hive", "sales", "orders")
        rels = {r["from_column"]: r["to_table"] for r in result["relationships"]}
        assert rels["customer_id"] == "customers"
        assert rels["product_id"] == "products"

    def test_table_relationships_none(self):
        result = srv.get_table_relationships("hive", "sales", "customers")
        assert result["relationship_count"] == 0


class TestForeignKeyParsing:
    def test_parse_fk_comment(self):
        rels = _parse_foreign_keys(
            [{"name": "customer_id", "comment": "FK to customers.customer_id"}]
        )
        assert rels == [{
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "customer_id",
            "type": "foreign_key",
        }]

    def test_parse_references_comment(self):
        rels = _parse_foreign_keys(
            [{"name": "product_id", "comment": "References products(product_id)"}]
        )
        assert rels[0]["to_table"] == "products"

    def test_ignore_non_fk_comment(self):
        rels = _parse_foreign_keys(
            [{"name": "region", "comment": "Geographic sales region"}]
        )
        assert rels == []