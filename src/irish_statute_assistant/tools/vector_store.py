from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from irish_statute_assistant.config import Config

_COLLECTION_NAME = "irish_statutes"


class VectorStore:
    def __init__(self, config: Config, embeddings=None) -> None:
        self._persist_directory = config.chroma_db_path
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self._embedding_function = embeddings
        self._chroma = Chroma(
            collection_name=_COLLECTION_NAME,
            persist_directory=self._persist_directory,
            embedding_function=self._embedding_function,
        )

    def is_populated(self) -> bool:
        return self._chroma._collection.count() > 0

    def add_sections(self, sections: list[dict]) -> None:
        try:
            self._chroma._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass  # collection does not yet exist — nothing to delete
        self._chroma = Chroma(
            collection_name=_COLLECTION_NAME,
            persist_directory=self._persist_directory,
            embedding_function=self._embedding_function,
        )
        if sections:
            self._chroma.add_texts(
                texts=[s["page_content"] for s in sections],
                metadatas=[
                    {
                        "title": s["title"],
                        "url": s["url"],
                        "section_index": s["section_index"],
                    }
                    for s in sections
                ],
            )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        docs = self._chroma.similarity_search(query, k=top_k)
        return [
            {
                "page_content": doc.page_content,
                "title": doc.metadata["title"],
                "url": doc.metadata["url"],
                "section_index": int(doc.metadata["section_index"]),
            }
            for doc in docs
        ]
