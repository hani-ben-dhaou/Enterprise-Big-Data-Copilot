"""
Orchestrator — LangGraph-based pipeline controller.

Graph structure:

  START
    │
    ▼
  [retrieve_rag]      ← RAG retrieval
    │
    ▼
  [retrieve_schema]   ← MCP schema lookup
    │
    ▼
  [generate_sql]      ← SQL Agent call
    │
    ▼
  [validate_sql]      ← Validator check
    │
    ├──(invalid + retries left)──► [generate_sql]   (loop back)
    │
    ▼
  [execute_sql]       ← Optional best-effort execution (MCP → Trino)
    │
    ▼
  [format_response]   ← Response Formatter
    │
    ▼
  END

All state is carried in PipelineState and mutated by each node.
Nodes are pure functions — easy to test in isolation.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.sql_agent import SQLAgent
from app.core.config import get_settings
from app.core.exceptions import OrchestratorError
from app.core.logging import get_logger
from app.core.models import (
    CopilotResponse,
    PipelineState,
    ValidationStatus,
)
from app.formatter.response_formatter import ResponseFormatter
from app.mcp.client import MCPClient
from app.rag.retriever import RAGRetriever
from app.validator.sql_validator import SQLValidator

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Node functions — each receives PipelineState, returns dict patch
# ---------------------------------------------------------------------------

def node_retrieve_rag(state: PipelineState) -> dict:
    """Node 1: retrieve relevant documentation chunks."""
    logger.info("node_retrieve_rag", question=state.question[:60])
    from app.core.models import RAGContext

    try:
        retriever = RAGRetriever()
        rag_context = retriever.retrieve(state.question) or RAGContext()
    except Exception as exc:
        logger.warning("rag_retrieval_failed", error=str(exc))
        rag_context = RAGContext()  # degrade gracefully
    return {"rag_context": rag_context}


def node_retrieve_schema(state: PipelineState) -> dict:
    """Node 2: retrieve schema metadata via MCP."""
    logger.info("node_retrieve_schema", question=state.question[:60])
    client = MCPClient()
    try:
        schema_context = client.get_schema_context(state.question)
    except Exception as exc:
        logger.warning("mcp_retrieval_failed", error=str(exc))
        from app.core.models import SchemaContext
        schema_context = SchemaContext()  # degrade gracefully
    return {"schema_context": schema_context}


def node_generate_sql(state: PipelineState) -> dict:
    """Node 3: generate (or regenerate) Trino SQL."""
    agent = SQLAgent()
    logger.info(
        "node_generate_sql",
        attempt=state.regeneration_count + 1,
        question=state.question[:60],
    )

    # A non-empty validation result means we already generated once, so any
    # further call is a regeneration attempt.
    is_regeneration = (
        state.validation_result is not None and state.generated_sql is not None
    )

    try:
        if is_regeneration:
            generated = agent.regenerate(
                question=state.question,
                failed_sql=state.generated_sql.sql,
                errors=state.validation_result.errors,
                schema_context=state.schema_context,  # type: ignore[arg-type]
            )
        else:
            generated = agent.generate(
                question=state.question,
                schema_context=state.schema_context,  # type: ignore[arg-type]
                rag_context=state.rag_context,  # type: ignore[arg-type]
            )
    except Exception as exc:
        logger.error("sql_generation_failed", error=str(exc))
        return {"error": str(exc)}

    # LangGraph state is immutable between nodes: counters must be propagated
    # through the returned patch, never mutated in place.
    return {
        "generated_sql": generated,
        "regeneration_count": state.regeneration_count + int(is_regeneration),
    }


def node_validate_sql(state: PipelineState) -> dict:
    """Node 4: validate the generated SQL."""
    if state.error or state.generated_sql is None:
        # LangGraph forbids empty writes; write a null validation explicitly.
        return {"validation_result": None}

    validator = SQLValidator()
    result = validator.validate(
        sql=state.generated_sql.sql,
        schema_context=state.schema_context,  # type: ignore[arg-type]
    )
    return {"validation_result": result}


def node_execute_sql(state: PipelineState) -> dict:
    """
    Node 5 (optional): execute the validated query, best-effort.

    Runs only when execution is enabled and the SQL validated cleanly.
    Execution failures are recorded as warnings — the pipeline never crashes
    because the data platform is unavailable.
    """
    if not settings.enable_sql_execution:
        return {"execution_result": None}

    if (
        state.generated_sql is None
        or state.validation_result is None
        or not state.validation_result.is_valid
    ):
        return {"execution_result": None}

    client = MCPClient()
    try:
        result = client.execute_sql(
            state.generated_sql.sql,
            limit=settings.mcp_default_limit,
        )
    except Exception as exc:
        logger.warning("sql_execution_failed", error=str(exc))
        return {
            "execution_result": {
                "status": "failed",
                "error": str(exc),
            }
        }

    if "error" in result:
        logger.warning("sql_execution_rejected", error=result.get("error"))
        return {
            "execution_result": {
                "status": "failed",
                "error": result.get("error"),
            }
        }

    summary = {
        "status": "ok",
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "row_count": result.get("row_count", len(result.get("rows", []))),
        "truncated": bool(result.get("truncated", False)),
    }
    logger.info(
        "sql_executed",
        row_count=summary["row_count"],
        truncated=summary["truncated"],
    )
    return {"execution_result": summary}


def node_format_response(state: PipelineState) -> dict:
    """Node 5: assemble the final CopilotResponse."""
    formatter = ResponseFormatter()
    response = formatter.format(state)
    return {"final_response": response}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_regenerate(
    state: PipelineState,
) -> Literal["generate_sql", "execute_sql", "format_response"]:
    """
    After validation: decide whether to retry, execute, or format.
    """
    max_attempts = settings.max_regeneration_attempts

    if state.error:
        return "format_response"

    if state.validation_result is None:
        return "format_response"

    if not state.validation_result.is_valid:
        if state.regeneration_count < max_attempts:
            logger.info(
                "regeneration_triggered",
                attempt=state.regeneration_count + 1,
                max=max_attempts,
            )
            return "generate_sql"
        return "format_response"

    return "execute_sql"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("retrieve_rag", node_retrieve_rag)
    graph.add_node("retrieve_schema", node_retrieve_schema)
    graph.add_node("generate_sql", node_generate_sql)
    graph.add_node("validate_sql", node_validate_sql)
    graph.add_node("execute_sql", node_execute_sql)
    graph.add_node("format_response", node_format_response)

    graph.add_edge(START, "retrieve_rag")
    graph.add_edge("retrieve_rag", "retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        should_regenerate,
        {
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql",
            "format_response": "format_response",
        },
    )

    graph.add_edge("execute_sql", "format_response")
    graph.add_edge("format_response", END)

    return graph


# ---------------------------------------------------------------------------
# Public Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Main entry point for the copilot pipeline.

    Usage:
        orchestrator = Orchestrator()
        response = orchestrator.run("Show me top 10 customers by revenue")
    """

    def __init__(self) -> None:
        graph = _build_graph()
        self._app = graph.compile()
        logger.info("orchestrator_initialized")

    def run(self, question: str) -> CopilotResponse:
        """
        Execute the full pipeline for a user question.

        Args:
            question: Natural language query from the user.

        Returns:
            CopilotResponse with SQL, explanation, and metadata.
        """
        logger.info("pipeline_start", question=question[:80])

        initial_state = PipelineState(question=question)

        try:
            final_state_dict = self._app.invoke(initial_state)
        except Exception as exc:
            logger.error("pipeline_crashed", error=str(exc))
            raise OrchestratorError(f"Pipeline failed: {exc}") from exc

        # LangGraph returns a dict; extract the PipelineState
        if isinstance(final_state_dict, dict):
            final_state = PipelineState(**final_state_dict)
        else:
            final_state = final_state_dict

        if final_state.final_response is None:
            raise OrchestratorError("Pipeline completed with no response.")

        logger.info(
            "pipeline_complete",
            confidence=final_state.final_response.confidence,
        )

        return final_state.final_response