"""
RAG Retriever — fetches the most relevant documentation chunks for a query.

Returns a RAGContext consumed by the SQL Agent.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.exceptions import RAGError
from app.core.logging import get_logger
from app.core.models import DocumentChunk, RAGContext
from app.rag.ingestor import get_vector_store

logger = get_logger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    Stateless retriever.  Instantiated once and reused across requests.
    """

    def __init__(self) -> None:
        self._store = get_vector_store()

    def retrieve(self, query: str) -> RAGContext:
        """
        Retrieve the top-k relevant document chunks for the given query.

        ``similarity_search_with_relevance_scores`` is used directly because
        ``as_retriever(search_type="similarity_score_threshold")`` discards the
        scores from the returned Document metadata in langchain-core.
        Results below the configured threshold are filtered out here.

        Args:
            query: Natural language question from the user.

        Returns:
            RAGContext containing ranked DocumentChunk objects.
        """
        try:
            docs_with_scores: list[tuple[Document, float]] = (
                self._store.similarity_search_with_relevance_scores(
                    query, k=settings.rag_top_k
                )
            )
            logger.info(
                "rag_retrieved",
                query=query[:80],
                num_docs=len(docs_with_scores),
            )
        except Exception as exc:
            raise RAGError(f"RAG retrieval failed: {exc}") from exc

        threshold = settings.rag_score_threshold
        chunks = [
            DocumentChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=round(score, 4),
                metadata=doc.metadata,
            )
            for doc, score in docs_with_scores
            if score >= threshold
        ]

        return RAGContext(chunks=chunks)