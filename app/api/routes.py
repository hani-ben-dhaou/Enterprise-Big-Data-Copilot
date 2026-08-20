"""
FastAPI route definitions for the Big Data Copilot API.

Endpoints:
  POST /api/v1/query         — main SQL generation endpoint
  GET  /api/v1/schema        — list available tables
  GET  /api/v1/health        — health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.exceptions import OrchestratorError
from app.core.logging import get_logger
from app.core.models import CopilotResponse
from app.mcp.catalog import get_catalog_service
from app.orchestrator.orchestrator import Orchestrator

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")

# Singleton orchestrator — shared across requests
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        examples=["Show me the top 10 customers by total revenue last month"],
    )


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"


class SchemaListResponse(BaseModel):
    catalogs: list[str]
    schemas: dict[str, list[str]]
    tables: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=CopilotResponse,
    summary="Convert natural language to Trino SQL",
    response_description="Validated Trino SQL with explanation and metadata",
)
def query(request: QueryRequest) -> CopilotResponse:
    """
    Main endpoint: accepts a natural language question and returns
    a schema-aware, validated Trino SQL query.

    Implemented as a *sync* handler: the pipeline blocks on LLM / Trino / MCP
    I/O, so FastAPI runs it in a worker thread instead of stalling the event
    loop for every request.
    """
    logger.info("api_query_received", question=request.question[:80])

    try:
        orchestrator = get_orchestrator()
        response = orchestrator.run(request.question)
    except OrchestratorError as exc:
        logger.error("api_orchestrator_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("api_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Please try again.",
        )

    return response


@router.get(
    "/schema",
    response_model=SchemaListResponse,
    summary="List available catalogs, schemas, and tables",
)
def list_schema() -> SchemaListResponse:
    """Return the full schema catalog available to the SQL Agent."""
    catalog = get_catalog_service()

    catalogs = catalog.list_catalogs()
    schemas: dict[str, list[str]] = {}
    tables: dict[str, list[str]] = {}

    for cat in catalogs:
        cat_schemas = catalog.list_schemas(cat)
        schemas[cat] = cat_schemas
        for schema in cat_schemas:
            key = f"{cat}.{schema}"
            tables[key] = catalog.list_tables(cat, schema)

    return SchemaListResponse(catalogs=catalogs, schemas=schemas, tables=tables)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")