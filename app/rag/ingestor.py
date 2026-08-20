"""
RAG Module — Document ingestion and embedding.

Responsible for:
- Loading documentation (Trino / Hive / Iceberg)
- Chunking and embedding documents
- Storing vectors in Qdrant
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model,
    )


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _ensure_collection(client: QdrantClient, vector_size: int = 1024) -> None:
    """Create Qdrant collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("qdrant_collection_created", name=settings.qdrant_collection)


_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".json"}


def _load_pdf(path: Path) -> list[Document]:
    """
    Extract text from a PDF using PyMuPDF (pymupdf).

    Falls back to an empty list with a warning if the library is missing or
    the file cannot be parsed, so ingestion never crashes on one bad file.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("pdf_loader_unavailable", path=str(path))
        return []

    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:
        logger.warning("pdf_parse_failed", path=str(path), error=str(exc))
        return []

    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_documents(docs_dir: Path) -> list[Document]:
    """
    Load all supported documentation files (.md/.txt/.rst/.json + .pdf).

    Plain text files use TextLoader; PDFs are parsed with PyMuPDF.
    Unsupported files are skipped with a warning instead of being read as
    binary text (which used to produce garbage chunks for PDFs).
    """
    documents: list[Document] = []

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_EXTENSIONS:
            try:
                documents.extend(
                    TextLoader(str(path), encoding="utf-8").load()
                )
            except Exception as exc:
                logger.warning("doc_load_failed", path=str(path), error=str(exc))
        elif suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            logger.warning("skipping_unsupported_doc", path=str(path))

    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_documentation(docs_dir: str | Path = "docs") -> int:
    """
    Full ingestion pipeline: load → chunk → embed → store.
    Returns the number of chunks stored.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    logger.info("ingestion_started", docs_dir=str(docs_path))

    raw_docs = _load_documents(docs_path)
    logger.info("docs_loaded", count=len(raw_docs))

    chunks = _chunk_documents(raw_docs)
    logger.info("chunks_created", count=len(chunks))

    client = _get_qdrant_client()
    _ensure_collection(client)

    embeddings = _get_embeddings()
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.qdrant_collection,
        url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
    )

    logger.info("ingestion_complete", chunks_stored=len(chunks))
    return len(chunks)


def get_vector_store() -> QdrantVectorStore:
    """Return a retrieval-ready vector store instance."""
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=_get_embeddings(),
    )