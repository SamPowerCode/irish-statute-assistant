from pydantic_settings import BaseSettings


class Config(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    evaluator_pass_threshold: float = 0.7
    max_refinement_rounds: int = 2
    max_retries: int = 3
    token_budget_per_query: int = 4000
    rate_limit_delay: float = 1.0

    model_config = {"env_file": ".env"}
