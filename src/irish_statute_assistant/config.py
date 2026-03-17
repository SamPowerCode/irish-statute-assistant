from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings

from irish_statute_assistant.llm import _DEFAULT_MODELS

_PROVIDER_KEY_MAP = {
    "anthropic": "anthropic_api_key",
    "openai":    "openai_api_key",
    "google":    "google_api_key",
    "groq":      "groq_api_key",
}


class Config(BaseSettings):
    anthropic_api_key: str = ""
    model_name: str = ""
    llm_provider: Literal["anthropic", "openai", "google", "groq"] = "anthropic"
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0
    # Vector store
    vector_store_backend: Literal["chroma", "qdrant"] = "chroma"
    chroma_db_path: str = "./data/chroma"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    index_categories: list[str] = [
        "employment", "housing", "family", "criminal", "contract",
        "personal injury", "planning", "company", "tax", "consumer",
    ]
    acts_per_category: int = 5

    model_config = {"env_file": ".env"}

    @model_validator(mode="after")
    def check_provider_and_set_model(self) -> "Config":
        # Set default model if not explicitly provided
        if not self.model_name:
            self.model_name = _DEFAULT_MODELS[self.llm_provider]
        # Validate that the required API key for the chosen provider is set
        key_field = _PROVIDER_KEY_MAP[self.llm_provider]
        if not getattr(self, key_field):
            raise ValueError(
                f"{key_field.upper()} is required when LLM_PROVIDER={self.llm_provider!r}"
            )
        return self
