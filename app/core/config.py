"""
Core configuration — loaded once at startup via pydantic-settings.
All modules import from here; never import os.environ directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "mxbai-embed-large:latest"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "Copilot-collection"

    # MCP
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8001
    # transport used by the MCPClient: "sse" (real MCP protocol) | "inprocess"
    mcp_transport: str = "inprocess"
    # metadata backend for the MCP server/tools: "inmemory" | "trino"
    mcp_metadata_source: str = "inmemory"
    # allow/deny table-scanning query tools (get_table_sample/stats)
    mcp_profiling_enabled: bool = True
    # allow/deny the execute_sql tool entirely
    mcp_query_execution_enabled: bool = True
    # default row cap for MCP query execution
    mcp_default_limit: int = 100

    # Trino
    trino_host: str = "localhost"
    trino_port: int = 8080
    trino_user: str = "Admin"
    trino_catalog: str = "hive"
    trino_schema: str = "default"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.6

    # SQL Agent
    max_regeneration_attempts: int = 3
    # timeout (seconds) for LLM generation calls; prevents a hung Ollama
    # from holding up the API forever
    llm_timeout: int = 60

    # Query execution (best-effort; graceful when the platform is unavailable)
    enable_sql_execution: bool = True

    # LangSmith / LangGraph tracing (optional, free tier at smith.langchain.com)
    # langchain/langgraph read these from the process environment; the Settings
    # fields exist so the env-file values are accepted and documented in one place.
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "copilot"
    langchain_endpoint: str = "https://api.smith.langchain.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()