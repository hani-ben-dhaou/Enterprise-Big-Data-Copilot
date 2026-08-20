"""
RAG tests: PDF/text loading semantics and retriever threshold filtering.

The retriever threshold tests bypass Qdrant with a fake store; the loader
tests use real small files on disk (PyMuPDF can both read and write PDFs).
"""

from langchain_core.documents import Document

from app.core.exceptions import RAGError
from app.core.models import RAGContext
from app.rag.ingestor import _load_documents, _load_pdf
from app.rag.retriever import RAGRetriever


def test_load_pdf_extracts_text(tmp_path):
    import fitz

    pdf_path = tmp_path / "guide.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Trino connector configuration reference text.")
    doc.save(pdf_path)
    doc.close()

    loaded = _load_pdf(pdf_path)
    assert len(loaded) == 1
    assert "Trino connector" in loaded[0].page_content
    assert loaded[0].metadata["source"] == "guide.pdf"


def test_unsupported_files_skipped(tmp_path):
    (tmp_path / "notes.docx").write_bytes(b"not real docx")
    (tmp_path / "README.md").write_text("# Trino Docs\nUse hive catalog.", encoding="utf-8")

    loaded = _load_documents(tmp_path)
    assert len(loaded) == 1
    assert "hive catalog" in loaded[0].page_content


def test_binary_file_with_unknown_suffix_not_loaded(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\nbinary data")
    assert _load_documents(tmp_path) == []


class _FakeStore:
    """Minimal vector store stub exposing the retriever's one method."""

    def __init__(self, results: list[tuple[Document, float]]):
        self._results = results
        self.calls = 0

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        self.calls += 1
        return self._results


def _fake_retriever(store: _FakeStore) -> RAGRetriever:
    # Bypass __init__ so no Qdrant connection is attempted.
    retriever = RAGRetriever.__new__(RAGRetriever)
    retriever._store = store
    return retriever


def test_retriever_filters_below_threshold():
    store = _FakeStore(
        [
            (Document(page_content="perfect match", metadata={"source": "a.md"}), 0.95),
            (Document(page_content="weak match", metadata={"source": "b.md"}), 0.20),
        ]
    )
    retriever = _fake_retriever(store)

    ctx = retriever.retrieve("trino config")
    assert isinstance(ctx, RAGContext)
    assert store.calls == 1
    assert len(ctx.chunks) == 1
    assert ctx.chunks[0].content == "perfect match"
    assert ctx.chunks[0].score == 0.95


def test_retriever_returns_empty_when_all_below_threshold():
    store = _FakeStore(
        [
            (Document(page_content="low", metadata={"source": "a.md"}), 0.1),
        ]
    )
    ctx = _fake_retriever(store).retrieve("anything")
    assert ctx.chunks == []


def test_retriever_propagates_store_failure():
    class _BrokenStore:
        def similarity_search_with_relevance_scores(self, query, k):
            raise ConnectionError("qdrant down")

    retriever = _fake_retriever(_BrokenStore())  # type: ignore[arg-type]
    import pytest

    with pytest.raises(RAGError):
        retriever.retrieve("anything")