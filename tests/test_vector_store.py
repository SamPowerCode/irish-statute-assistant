import logging
import pytest
from unittest.mock import patch
from irish_statute_assistant.config import Config
from irish_statute_assistant.tools.vector_store import VectorStore, get_vector_store
from irish_statute_assistant.tools.qdrant_vector_store import QdrantVectorStore


class _StubEmbeddings:
    """4-dim fixed-vector stub. Avoids downloading sentence-transformer models in tests."""
    dim = 4

    def embed_documents(self, texts):
        return [[1.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [1.0] * self.dim


def make_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(chroma_db_path=str(tmp_path / "chroma"))
    return VectorStore(config, embeddings=_StubEmbeddings())


def test_is_populated_false_on_fresh_store(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    assert store.is_populated() is False


def test_is_populated_true_after_add_sections(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "An employer must give notice.", "title": "Employment Act 2001",
         "url": "https://example.com/act1", "section_index": 0},
    ])
    assert store.is_populated() is True


def test_is_populated_false_after_add_sections_empty_list(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Some text.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 0},
    ])
    store.add_sections([])  # wipe
    assert store.is_populated() is False


def test_search_returns_correct_keys(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Tenant rights.", "title": "Housing Act 1992",
         "url": "https://example.com/housing", "section_index": 2},
    ])
    results = store.search("tenant", top_k=5)
    assert len(results) == 1
    assert results[0]["page_content"] == "Tenant rights."
    assert results[0]["title"] == "Housing Act 1992"
    assert results[0]["url"] == "https://example.com/housing"
    assert results[0]["section_index"] == 2


def test_search_returns_section_index_as_int(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Section text.", "title": "Act B",
         "url": "https://example.com/b", "section_index": 7},
    ])
    results = store.search("text", top_k=5)
    assert isinstance(results[0]["section_index"], int)
    assert results[0]["section_index"] == 7


def test_add_sections_wipes_on_second_call(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.add_sections([
        {"page_content": "Old content.", "title": "Old Act",
         "url": "https://example.com/old", "section_index": 0},
    ])
    store.add_sections([
        {"page_content": "New content.", "title": "New Act",
         "url": "https://example.com/new", "section_index": 0},
    ])
    results = store.search("content", top_k=10)
    assert len(results) == 1
    assert results[0]["title"] == "New Act"


# ---------------------------------------------------------------------------
# get_vector_store factory tests
# ---------------------------------------------------------------------------

def test_factory_returns_chroma_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(chroma_db_path=str(tmp_path / "chroma"))
    store = get_vector_store(config, embeddings=_StubEmbeddings())
    assert isinstance(store, VectorStore)


def test_factory_returns_qdrant_when_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(vector_store_backend="qdrant")
    store = get_vector_store(config, embeddings=_StubEmbeddings())
    assert isinstance(store, QdrantVectorStore)


# ---------------------------------------------------------------------------
# QdrantVectorStore tests (in-memory — no server needed)
# ---------------------------------------------------------------------------

def make_qdrant_store(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    config = Config(vector_store_backend="qdrant")  # qdrant_url="" → in-memory
    return QdrantVectorStore(config, embeddings=_StubEmbeddings())


def test_qdrant_is_populated_false_on_fresh_store(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    assert store.is_populated() is False


def test_qdrant_is_populated_true_after_add_sections(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    store.add_sections([
        {"page_content": "An employer must give notice.", "title": "Employment Act 2001",
         "url": "https://example.com/act1", "section_index": 0},
    ])
    assert store.is_populated() is True


def test_qdrant_is_populated_false_after_wipe(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    store.add_sections([
        {"page_content": "Some text.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 0},
    ])
    store.add_sections([])
    assert store.is_populated() is False


def test_qdrant_search_returns_correct_keys(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    store.add_sections([
        {"page_content": "Tenant rights.", "title": "Housing Act 1992",
         "url": "https://example.com/housing", "section_index": 2},
    ])
    results = store.search("tenant", top_k=5)
    assert len(results) == 1
    assert results[0]["page_content"] == "Tenant rights."
    assert results[0]["title"] == "Housing Act 1992"
    assert results[0]["url"] == "https://example.com/housing"
    assert results[0]["section_index"] == 2


def test_qdrant_search_returns_section_index_as_int(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    store.add_sections([
        {"page_content": "Section text.", "title": "Act B",
         "url": "https://example.com/b", "section_index": 7},
    ])
    results = store.search("text", top_k=5)
    assert isinstance(results[0]["section_index"], int)
    assert results[0]["section_index"] == 7


def test_qdrant_add_sections_wipes_on_second_call(monkeypatch):
    store = make_qdrant_store(monkeypatch)
    store.add_sections([
        {"page_content": "Old content.", "title": "Old Act",
         "url": "https://example.com/old", "section_index": 0},
    ])
    store.add_sections([
        {"page_content": "New content.", "title": "New Act",
         "url": "https://example.com/new", "section_index": 0},
    ])
    results = store.search("content", top_k=10)
    assert len(results) == 1
    assert results[0]["title"] == "New Act"


def test_qdrant_search_logs_warning_on_exception(monkeypatch, caplog):
    store = make_qdrant_store(monkeypatch)
    with patch.object(store._embeddings, "embed_query", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING, logger="irish_statute_assistant.tools.qdrant_vector_store"):
            result = store.search("anything")
    assert result == []
    assert any("boom" in r.message for r in caplog.records)
