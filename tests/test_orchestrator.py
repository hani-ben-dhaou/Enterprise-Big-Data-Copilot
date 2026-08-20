"""
Orchestrator / LangGraph regression tests.

These exercise the real compiled graph (real langgraph) with a stubbed
SQLAgent so no Ollama/Qdrant/Trino is required.

Regression coverage:
  - the regeneration loop terminates after ``max_regeneration_attempts``
    (LangGraph state is immutable between nodes: counters must be propagated
    through node return patches, never mutated in place);
  - invalid SQL triggers a regeneration, and the regenerated (valid) SQL is
    used downstream;
  - every node always writes at least one state channel (no ``InvalidUpdateError``).
"""

import pytest

from app.core.models import GeneratedSQL
from app.orchestrator import orchestrator
from app.orchestrator.orchestrator import _build_graph
from app.orchestrator.orchestrator import settings as orch_settings


class _CountingAgent:
    """Fake SQLAgent that counts calls and returns configurable SQL."""

    def __init__(self, sqls):
        self.sqls = sqls
        self.calls = 0

    def _sql(self):
        sql = self.sqls[min(self.calls - 1, len(self.sqls) - 1)]
        return GeneratedSQL(sql=sql, explanation="stub", confidence=0.9)

    def generate(self, question, schema_context, rag_context=None):
        self.calls += 1
        return self._sql()

    def regenerate(self, question, failed_sql, errors, schema_context):
        return self.generate(question, schema_context)


_INVALID = "INSERT INTO hive.sales.orders VALUES (1)"
_VALID = "SELECT count(*) FROM hive.sales.orders"


def _stub_agent(monkeypatch, agent):
    monkeypatch.setattr(orchestrator, "SQLAgent", lambda *a, **k: agent)


def test_regeneration_loop_terminates(monkeypatch):
    """Always-invalid SQL must stop after 1 initial + max regeneration attempts."""
    agent = _CountingAgent([_INVALID])
    _stub_agent(monkeypatch, agent)

    final = _build_graph().compile().invoke({"question": "q"})

    assert agent.calls == orch_settings.max_regeneration_attempts + 1
    resp = final["final_response"]
    assert resp is not None
    assert any("regenerated" in w for w in resp.warnings)


def test_invalid_then_valid_regenerates_and_uses_regenerated_sql(monkeypatch):
    """Invalid first attempt must regenerate; the corrected SQL is executed."""
    agent = _CountingAgent([_INVALID, _VALID])
    _stub_agent(monkeypatch, agent)

    final = _build_graph().compile().invoke({"question": "q"})
    resp = final["final_response"]

    assert agent.calls == 2
    assert final["regeneration_count"] == 1
    assert resp.sql == _VALID


def test_valid_single_pass_no_regeneration(monkeypatch):
    """A valid first attempt must not regenerate."""
    agent = _CountingAgent([_VALID])
    _stub_agent(monkeypatch, agent)

    final = _build_graph().compile().invoke({"question": "q"})

    assert agent.calls == 1
    assert final["regeneration_count"] == 0
    assert final["final_response"].sql == _VALID


def test_generation_failure_cleans_up(monkeypatch):
    """If generation raises, the pipeline must still produce an error response
    instead of crashing with InvalidUpdateError."""

    class Boom:
        def generate(self, question, schema_context, rag_context=None):
            raise RuntimeError("llm down")

        def regenerate(self, question, failed_sql, errors, schema_context):
            raise RuntimeError("llm down")

    _stub_agent(monkeypatch, Boom())
    final = _build_graph().compile().invoke({"question": "q"})
    resp = final["final_response"]

    assert "An error occurred" in resp.explanation
    assert resp.confidence == 0.0


def test_execution_disabled_still_formats(monkeypatch):
    """With SQL execution disabled, a valid query must still return a response."""
    agent = _CountingAgent([_VALID])
    _stub_agent(monkeypatch, agent)

    monkeypatch.setattr(orch_settings, "enable_sql_execution", False)
    final = _build_graph().compile().invoke({"question": "q"})

    assert final["final_response"].sql == _VALID
    assert agent.calls == 1