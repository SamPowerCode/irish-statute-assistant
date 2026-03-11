from pydantic_settings import BaseSettings


class Config(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    index_categories: list[str] = [
        "employment", "housing", "family", "criminal", "contract",
        "personal injury", "planning", "company", "tax", "consumer",
    ]
    acts_per_category: int = 5

    model_config = {"env_file": ".env"}
