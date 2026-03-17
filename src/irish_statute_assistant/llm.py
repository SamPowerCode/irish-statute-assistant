"""LLM provider factory for Irish Statute Assistant."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from irish_statute_assistant.config import Config

# Single source of truth for default models per provider.
# Imported by config.py — do not duplicate this dict.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
}


def get_llm(config: Config, max_tokens: int) -> Any:
    """Return the LangChain chat model for the configured provider.

    Imports are lazy so missing optional packages only fail if that
    provider is actually selected.
    """
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.openai_api_key,
            max_tokens=max_tokens,
        )
    elif config.llm_provider == "google":
        # Google uses max_output_tokens, not max_tokens
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            max_output_tokens=max_tokens,
        )
    elif config.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.model_name,
            api_key=config.groq_api_key,
            max_tokens=max_tokens,
        )
    else:  # anthropic (default)
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=max_tokens,
        )
