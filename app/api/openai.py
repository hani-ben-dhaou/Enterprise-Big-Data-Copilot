"""
OpenAI-compatible API endpoints for Open WebUI integration.

Endpoints:
  GET  /v1/models           — list available models
  POST /v1/chat/completions — RAG-enhanced chat (no streaming)
"""

from __future__ import annotations

import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes import get_orchestrator
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.models import CopilotResponse
from app.rag.retriever import RAGRetriever

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# OpenAI-compatible request/response models (minimal subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = settings.ollama_model
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = 1024
    stream: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "copilot"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_rag_context(rag_context) -> str:
    """Format RAG context chunks into a single text block."""
    if not rag_context or not rag_context.chunks:
        return ""
    parts = [f"[Source: {c.source}]\n{c.content}" for c in rag_context.chunks]
    return "\n\n---\n\n".join(parts)


def _count_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Copilot pipeline routing — WebUI chat can answer from real executed SQL
# ---------------------------------------------------------------------------

_DATA_KEYWORDS = (
    "count", "how many", "how much", "number of", "total", "sum", "average",
    "avg", "top", "list", "show", "orders", "order", "customers", "customer",
    "revenue", "sales", "product", "products", "table", "tables", "schema",
    "schemas", "catalog", "catalogs", "query", "where", "max", "min",
    "group by", "join",
)


def _is_data_question(text: str) -> bool:
    """Heuristic: does this sound like a data/query question or small talk?"""
    lowered = text.lower().strip()
    if len(lowered) < 5:
        return False
    return any(kw in lowered for kw in _DATA_KEYWORDS)


def _fmt_cell(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _copilot_to_text(response: CopilotResponse) -> str:
    """Render a CopilotResponse as a plain chat answer (real results)."""
    lines: list[str] = []
    if response.explanation:
        lines.append(response.explanation.strip())

    execution = response.execution or {}
    if execution.get("status") == "ok":
        rows = execution.get("rows") or []
        columns = execution.get("columns") or (["result"] if rows else [])
        lines.append("")
        lines.append("Verified result (executed on Trino):")
        for row in rows:
            labels = ", ".join(f"{col}={_fmt_cell(v)}" for col, v in zip(columns, row))
            lines.append(f"  {labels}")
        lines.append(f"  ({len(rows)} row{'s' if len(rows) != 1 else ''} returned)")
    elif execution:
        lines.append("")
        lines.append(f"Execution status: {execution.get('status', 'unknown')}")

    if response.sql:
        lines.append("")
        lines.append(f"SQL used: {response.sql}")
    if response.warnings:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {w}" for w in response.warnings)

    return "\n".join(lines)


def _run_copilot_pipeline(question: str) -> str | None:
    """
    Run the full copilot pipeline (RAG -> schema -> SQL -> validate -> execute).
    Returns a chat-answer string, or None so callers can fall back to plain chat.
    """
    try:
        response = get_orchestrator().run(question)
    except Exception as exc:
        logger.warning("chat_pipeline_failed_fallback_to_chat", error=str(exc))
        return None
    if response is None or not response.sql:
        return None
    return _copilot_to_text(response)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelList)
def list_models():
    """
    Return available models (pulled from Ollama).

    Sync handler: the Ollama HTTP call blocks, so FastAPI runs it in a worker
    thread instead of stalling the event loop.
    """
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{settings.ollama_base_url}/api/tags")
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        now = int(time.time())
        models = [
            ModelInfo(id=m["name"], created=now, owned_by="ollama")
            for m in tags
        ]
    return ModelList(data=models)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest):
    """
    RAG-enhanced chat: retrieves relevant docs from Qdrant,
    augments the prompt, then delegates to Ollama.

    Sync handler for the same reason as list_models.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract the last user message
    user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        None,
    )
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message found")

    logger.info("chat_request", model=request.model, question=user_msg[:80])

    # Data question -> run the real copilot pipeline (grounded, executed SQL).
    if _is_data_question(user_msg):
        pipeline_answer = _run_copilot_pipeline(user_msg)
        if pipeline_answer is not None:
            logger.info("chat_pipeline_used", question=user_msg[:80])
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatChoice(message=ChatMessage(role="assistant", content=pipeline_answer))
                ],
                usage=UsageInfo(total_tokens=_count_tokens(pipeline_answer)),
            )
        logger.info("chat_pipeline_fallback_to_rag", question=user_msg[:80])

    # 1. Retrieve RAG context from Qdrant
    rag_context = None
    try:
        retriever = RAGRetriever()
        rag_context = retriever.retrieve(user_msg)
        logger.info("rag_retrieved", num_chunks=len(rag_context.chunks))
    except Exception as exc:
        logger.warning("rag_retrieval_failed", error=str(exc))

    # 2. Build messages for Ollama
    ollama_messages = []
    for m in request.messages:
        if m.role == "system":
            ollama_messages.append({"role": "system", "content": m.content})
        elif m.role == "user":
            ollama_messages.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            ollama_messages.append({"role": "assistant", "content": m.content})

    # Inject RAG context into system message
    rag_text = _format_rag_context(rag_context)
    if rag_text:
        system_content = (
            "You are a big data expert assistant. Use the following documentation "
            "context to answer the user's question. If the context doesn't contain "
            "relevant information, answer based on your general knowledge.\n\n"
            f"### Documentation Context:\n{rag_text}"
        )
        # Prepend as system message
        ollama_messages.insert(0, {"role": "system", "content": system_content})

    # 3. Call Ollama
    ollama_payload = {
        "model": request.model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": request.temperature or 0.7,
            "num_predict": request.max_tokens or 1024,
        },
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=ollama_payload,
            )
            resp.raise_for_status()
            ollama_result = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    # 4. Build OpenAI-compatible response
    answer = ollama_result.get("message", {}).get("content", "")
    prompt = "\n".join(m.get("content", "") for m in ollama_messages)

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=answer),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=_count_tokens(prompt),
            completion_tokens=_count_tokens(answer),
            total_tokens=_count_tokens(prompt) + _count_tokens(answer),
        ),
    )
