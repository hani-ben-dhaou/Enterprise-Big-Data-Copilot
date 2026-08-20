"""
SQL Validator — enforces safety and schema correctness.

Checks (in order):
  1. SELECT-only guard  (no DDL / DML / destructive ops)
  2. Trino SQL parse    (via sqlglot)
  3. Schema grounding   (referenced tables/columns exist in SchemaContext)
  4. Hardcoded guards   (DROP / DELETE / TRUNCATE / INSERT)

Returns a ValidationResult with status, warnings, and errors.
"""

from __future__ import annotations

import re
from typing import Sequence

import sqlglot
import sqlglot.expressions as exp

from app.core.logging import get_logger
from app.core.models import (
    SchemaContext,
    ValidationResult,
    ValidationStatus,
)

logger = get_logger(__name__)

# Keywords that are never allowed in V1
_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset([
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "MERGE", "REPLACE", "GRANT", "REVOKE",
    "SET", "EXECUTE", "CALL",
])

# Matches string literals ('...' with '' escapes), double-quoted identifiers,
# and comments — removed before the keyword scan so that forbidden words
# appearing in *data* (not statement syntax) never cause false positives.
_QUOTED_STRING_RE = re.compile(r"'(''|[^'])*'")
_DOUBLE_QUOTED_RE = re.compile(r'"[^"]*"')
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


class SQLValidator:
    """
    Validates a generated SQL string against safety rules and the
    known schema context.
    """

    def validate(
        self,
        sql: str,
        schema_context: SchemaContext,
    ) -> ValidationResult:
        """
        Run all validation checks.

        Returns ValidationResult with collected errors and warnings.
        The caller uses result.is_valid to decide whether to retry.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Forbidden keyword check (fast)
        kw_errors = self._check_forbidden_keywords(sql)
        errors.extend(kw_errors)

        # 2. Parse with sqlglot (dialect=trino)
        parsed = self._parse_sql(sql, errors)

        if parsed is not None:
            # 3. Must be a SELECT statement
            select_errors = self._check_is_select(parsed)
            errors.extend(select_errors)

            # 4. Schema grounding
            schema_errors = self._check_schema_grounding(parsed, schema_context)
            errors.extend(schema_errors)

            # 5. LIMIT check (warn only)
            if self._missing_limit(parsed):
                warnings.append(
                    "Query has no LIMIT clause — may return very large result sets."
                )

        if errors:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        logger.info(
            "validation_complete",
            status=status,
            errors=errors,
            warnings=warnings,
        )

        return ValidationResult(status=status, warnings=warnings, errors=errors)

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_literals_and_comments(sql: str) -> str:
        """Remove quoted strings, identifiers, and comments before scanning,
        so forbidden keywords inside data cannot trigger false positives."""
        s = _QUOTED_STRING_RE.sub("''", sql)
        s = _DOUBLE_QUOTED_RE.sub('""', s)
        s = _LINE_COMMENT_RE.sub("", s)
        s = _BLOCK_COMMENT_RE.sub("", s)
        return s

    @staticmethod
    def _check_forbidden_keywords(sql: str) -> list[str]:
        upper = SQLValidator._strip_literals_and_comments(sql).upper()
        found = [kw for kw in _FORBIDDEN_KEYWORDS if re.search(rf"\b{kw}\b", upper)]
        if found:
            return [f"Forbidden SQL keywords detected: {', '.join(found)}"]
        return []

    @staticmethod
    def _parse_sql(sql: str, errors: list[str]) -> exp.Expression | None:
        try:
            statements = sqlglot.parse(sql, dialect="trino")
            if not statements or statements[0] is None:
                errors.append("SQL could not be parsed (empty result).")
                return None
            return statements[0]
        except sqlglot.errors.ParseError as exc:
            errors.append(f"SQL parse error: {exc}")
            return None

    @staticmethod
    def _check_is_select(parsed: exp.Expression) -> list[str]:
        if not isinstance(parsed, exp.Select):
            return [
                f"Only SELECT statements are allowed. Got: {type(parsed).__name__}"
            ]
        return []

    @staticmethod
    def _check_schema_grounding(
        parsed: exp.Expression,
        schema_context: SchemaContext,
    ) -> list[str]:
        """
        Reject queries that reference tables or columns outside the schema context.

        Table references:
          - CTE names are always allowed (they are defined by the query itself).
          - Fully-qualified refs (catalog.schema.table) must match an FQN in
            context; unqualified refs fall back to a bare-table-name match.
        Column references:
          - SELECT/GROUP BY/ORDER BY aliases are allowed (they are defined by
            the query itself).
          - A known table qualifier is checked against that table's columns;
            unqualified columns are checked against the union of all columns.

        Out-of-context references are hard errors: the schema context is the
        only dataset the agent may read, so ungrounded refs are hallucinated
        (e.g. tpch.sf1.partsupp when only the tiny schema is in scope).
        """
        if not schema_context.tables:
            return []

        known_fqns: set[str] = set()
        known_tables: set[str] = set()
        table_columns: dict[str, set[str]] = {}
        all_columns: set[str] = set()
        for t in schema_context.tables:
            fqn = f"{t.catalog}.{t.schema_name}.{t.table_name}".lower()
            known_fqns.add(fqn)
            lower_t = t.table_name.lower()
            known_tables.add(lower_t)
            table_columns.setdefault(lower_t, set())
            for c in t.columns:
                cn = c.name.lower()
                table_columns[lower_t].add(cn)
                all_columns.add(cn)

        # Names defined *by the query itself* are not schema objects.
        cte_names: set[str] = {
            cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE)
        }
        aliases: set[str] = {
            a.alias.lower() for a in parsed.find_all(exp.Alias) if a.alias
        }

        errors: list[str] = []

        for table_ref in parsed.find_all(exp.Table):
            ref_name = table_ref.name.lower() if table_ref.name else ""
            if not ref_name or ref_name in cte_names:
                continue

            ref_parts = [
                p for p in [table_ref.catalog, table_ref.db, table_ref.name]
                if p
            ]
            ref_str = ".".join(ref_parts)

            # Fully-qualified refs must match a *known FQN*; a bare-name match
            # on the wrong catalog/schema (tpch.tiny.orders vs hive.sales.orders)
            # is exactly what we want to catch.
            if table_ref.catalog or table_ref.db:
                if ref_str.lower() not in known_fqns:
                    errors.append(
                        f"Table '{ref_str}' was not found in the provided schema context."
                    )
                continue

            # Unqualified ref: fall back to a bare-table-name match.
            if ref_name not in known_tables:
                errors.append(
                    f"Table '{ref_str}' was not found in the provided schema context."
                )

        for col in parsed.find_all(exp.Column):
            cname = col.name.lower() if col.name else ""
            if not cname or cname in aliases:
                continue
            tname = col.table.lower() if col.table else ""
            if tname in known_tables:
                if cname not in table_columns.get(tname, set()):
                    errors.append(
                        f"Column '{col.name}' was not found on table "
                        f"'{col.table}' in the provided schema context."
                    )
            elif not tname and cname not in all_columns:
                errors.append(
                    f"Column '{col.name}' was not found in the provided "
                    "schema context."
                )

        return errors

    @staticmethod
    def _missing_limit(parsed: exp.Expression) -> bool:
        """True if SELECT has no LIMIT/FETCH FIRST clause."""
        if not isinstance(parsed, exp.Select):
            return False
        return parsed.find(exp.Limit) is None