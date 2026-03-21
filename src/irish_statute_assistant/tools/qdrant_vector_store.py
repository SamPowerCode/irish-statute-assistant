"""Qdrant-backed vector store — drop-in replacement for VectorStore (Chroma)."""
from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from irish_statute_assistant.config import Config

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "irish_statutes"


class QdrantVectorStore:
    def __init__(self, config: Config, embeddings=None) -> None:
        if config.qdrant_url:
            self._client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key or None,
            )
        else:
            self._client = QdrantClient(":memory:")

        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self._embeddings = embeddings
        self._vector_size = len(self._embeddings.embed_query("test"))

    def is_populated(self) -> bool:
        try:
            info = self._client.get_collection(_COLLECTION_NAME)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def add_sections(self, sections: list[dict]) -> None:
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass

        self._client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )

        if sections:
            texts = [s["page_content"] for s in sections]
            vectors = self._embeddings.embed_documents(texts)
            points = [
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "page_content": s["page_content"],
                        "title": s["title"],
                        "url": s["url"],
                        "section_index": s["section_index"],
                    },
                )
                for idx, (s, vector) in enumerate(zip(sections, vectors))
            ]
            self._client.upsert(collection_name=_COLLECTION_NAME, points=points)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        try:
            vector = self._embeddings.embed_query(query)
            result = self._client.query_points(
                collection_name=_COLLECTION_NAME,
                query=vector,
                limit=top_k,
            )
            return [
                {
                    "page_content": r.payload["page_content"],
                    "title": r.payload["title"],
                    "url": r.payload["url"],
                    "section_index": int(r.payload["section_index"]),
                }
                for r in result.points
            ]
        except Exception as e:
            logger.warning("Qdrant search failed: %s", e)
            return []
