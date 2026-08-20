"""
API endpoint tests (FastAPI TestClient) for the V1 REST surface.

These tests only exercise the HTTP layer: the orchestrator and catalog service
are faked so nothing touches Ollama / Trino / MCP / Qdrant.
"""

from fastapi.testclient import TestClient

from app.api.routes import get_catalog_service, get_orchestrator
from app.core.exceptions import OrchestratorError
from app.core.models import CopilotResponse
from app.main import app

client = TestClient(app)


class _FakeOrchestrator:
    def __init__(self, response: CopilotResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error

    def run(self, question: str) -> CopilotResponse:
        if self.error is not None:
            raise self.error
        return self.response or CopilotResponse(
            question=question,
            sql="SELECT * FROM hive.sales.orders LIMIT 10",
            explanation="Fake orchestrator result.",
            confidence=0.9,
        )


class _FakeCatalog:
    def list_catalogs(self) -> list[str]:
        return ["hive"]

    def list_schemas(self, catalog: str) -> list[str]:
        return ["sales"]

    def list_tables(self, catalog: str, schema: str) -> list[str]:
        return ["orders"]


def test_health(monkeypatch):
    monkeypatch.setattr("app.api.routes.get_orchestrator", get_orchestrator)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_schema_lists_catalog(monkeypatch):
    monkeypatch.setattr("app.api.routes.get_catalog_service", lambda: _FakeCatalog())
    resp = client.get("/api/v1/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert body["catalogs"] == ["hive"]
    assert body["schemas"] == {"hive": ["sales"]}
    assert "hive.sales" in body["tables"]
    assert body["tables"]["hive.sales"] == ["orders"]


def test_query_success(monkeypatch):
    monkeypatch.setattr("app.api.routes.get_orchestrator", lambda: _FakeOrchestrator())
    resp = client.post("/api/v1/query", json={"question": "Show me recent orders"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"].startswith("SELECT")
    assert body["explanation"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_query_short_question_rejected():
    resp = client.post("/api/v1/query", json={"question": "hey"})
    assert resp.status_code == 422


def test_query_orchestrator_error_returns_500(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.get_orchestrator",
        lambda: _FakeOrchestrator(error=OrchestratorError("boom")),
    )
    resp = client.post("/api/v1/query", json={"question": "Show me recent orders"})
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_query_unexpected_error_returns_500(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.get_orchestrator",
        lambda: _FakeOrchestrator(error=RuntimeError("boom")),
    )
    resp = client.post("/api/v1/query", json={"question": "Show me recent orders"})
    assert resp.status_code == 500
    assert "Internal server error" in resp.json()["detail"]


def test_chat_completions_rejects_empty_messages():
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": []},
    )
    assert resp.status_code == 400


def test_chat_completions_rejects_no_user_message():
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Copilot-pipeline routing in the OpenAI-compatible chat endpoint
# ---------------------------------------------------------------------------


def _offline_chat(monkeypatch):
    """Keep the plain-chat fallback fully offline for tests."""

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "fallback reply"}}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, **kwargs):
            return _FakeResponse()

    class _FakeRag:
        def __init__(self, *args, **kwargs):
            pass

        def retrieve(self, question):
            class _Context:
                chunks = []

            return _Context()

    monkeypatch.setattr("app.api.openai.httpx.Client", _FakeClient)
    monkeypatch.setattr("app.api.openai.RAGRetriever", _FakeRag)


def test_data_question_heuristic():
    from app.api.openai import _is_data_question

    assert _is_data_question("How many orders are there in tpch.tiny.orders?")
    assert _is_data_question("show me the top 10 customers by revenue")
    assert _is_data_question("List all tables in the sales schema")
    assert not _is_data_question("hello, how are you?")
    assert not _is_data_question("What is the copilot?")


def test_chat_data_question_uses_pipeline(monkeypatch):
    fake = _FakeOrchestrator(
        response=CopilotResponse(
            question="how many orders",
            sql="SELECT count(*) FROM tpch.tiny.orders",
            explanation="Counting the orders in tpch.tiny.orders.",
            confidence=0.9,
            execution={
                "status": "ok",
                "columns": ["_col0"],
                "rows": [[15000]],
                "row_count": 1,
                "truncated": False,
            },
        )
    )
    monkeypatch.setattr("app.api.openai.get_orchestrator", lambda: fake)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "how many orders are there?"}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert "_col0=15000" in content
    assert "SELECT count(*) FROM tpch.tiny.orders" in content


def test_chat_casual_message_skips_pipeline(monkeypatch):
    _offline_chat(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("pipeline must not run for casual chat")

    fake = _FakeOrchestrator()
    fake.run = boom
    monkeypatch.setattr("app.api.openai.get_orchestrator", lambda: fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello, how are you?"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "fallback reply"


def test_chat_pipeline_error_falls_back_to_chat(monkeypatch):
    _offline_chat(monkeypatch)
    monkeypatch.setattr(
        "app.api.openai.get_orchestrator",
        lambda: _FakeOrchestrator(error=OrchestratorError("backend down")),
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "how many orders are there?"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "fallback reply"