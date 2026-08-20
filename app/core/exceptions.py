"""Domain exceptions for the copilot system."""


class CopilotError(Exception):
    """Base error for all copilot exceptions."""


class RAGError(CopilotError):
    """Raised when RAG retrieval fails."""


class MCPError(CopilotError):
    """Raised when MCP schema retrieval fails."""


class SQLGenerationError(CopilotError):
    """Raised when the SQL agent cannot generate a valid query."""


class ValidationError(CopilotError):
    """Raised when the validator rejects a query after max retries."""


class OrchestratorError(CopilotError):
    """Raised for pipeline-level failures."""