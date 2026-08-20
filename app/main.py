"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.openai import router as openai_router
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("startup", model=settings.ollama_model, env="v1")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Enterprise Big Data Copilot",
    description=(
        "AI-powered Trino SQL generation using RAG + MCP + LangGraph. "
        "Convert natural language questions into validated, schema-aware SQL queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(openai_router)