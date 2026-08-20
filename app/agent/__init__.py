# Lazy import — SQLAgent requires langchain_ollama (installed at runtime with Ollama)
# Do not import SQLAgent at module level to keep tests fast without full deps.

__all__ = ["SQLAgent"]


def __getattr__(name: str):
    if name == "SQLAgent":
        from app.agent.sql_agent import SQLAgent
        return SQLAgent
    raise AttributeError(f"module 'app.agent' has no attribute {name!r}")