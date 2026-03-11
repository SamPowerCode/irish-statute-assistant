import logging
from unittest.mock import MagicMock, patch
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.models.schemas import ResearcherOutput, ActSection
from irish_statute_assistant.tools.session_cache import SessionCache


def make_researcher_with_mocks(search_results, sections_per_act):
    cache = SessionCache()
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent._cache = cache

    # _vector_store required by run() after the vector-store update.
    # Set is_populated=False so existing tests continue to exercise the live-fetch path.
    from unittest.mock import MagicMock
    mock_store = MagicMock()
    mock_store.is_populated.return_value = False
    agent._vector_store = mock_store

    mock_search = MagicMock(return_value=search_results)
    mock_fetch = MagicMock(return_value=sections_per_act)
    agent._search = mock_search
    agent._fetch = mock_fetch
    return agent


def test_researcher_returns_researcher_output():
    agent = make_researcher_with_mocks(
        search_results=[{"title": "Act A", "url": "https://example.com/act-a"}],
        sections_per_act=["Section 1 text"],
    )
    result = agent.run(query="personal injury time limit")
    assert isinstance(result, ResearcherOutput)
    assert len(result.acts) >= 1


def test_researcher_includes_title_and_url():
    agent = make_researcher_with_mocks(
        search_results=[{"title": "Statute of Limitations Act 1957", "url": "https://example.com/1957"}],
        sections_per_act=["You must bring an action within 6 years."],
    )
    result = agent.run(query="limitation of actions")
    assert result.acts[0].title == "Statute of Limitations Act 1957"
    assert result.acts[0].url == "https://example.com/1957"


def test_researcher_raises_on_no_results():
    agent = make_researcher_with_mocks(search_results=[], sections_per_act=[])
    import pytest
    with pytest.raises(ValueError, match="No Acts found"):
        agent.run(query="completely unknown legal topic xyz")


def make_researcher_with_vector_store(mock_store, search_results=None, sections_per_act=None):
    """Create a ResearcherAgent with a mocked VectorStore and mocked live-fetch."""
    from irish_statute_assistant.config import Config
    from irish_statute_assistant.tools.session_cache import SessionCache
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test")

    with patch("irish_statute_assistant.agents.researcher.VectorStore", return_value=mock_store):
        config = Config()
        cache = SessionCache()
        agent = ResearcherAgent(config, cache)

    # Also wire live-fetch mocks for fallback tests
    if search_results is not None:
        agent._search = MagicMock(return_value=search_results)
    if sections_per_act is not None:
        agent._fetch = MagicMock(return_value=sections_per_act)
    return agent


def test_researcher_uses_vector_store_when_populated():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    mock_store.search.return_value = [
        {"page_content": "Workers must receive notice.", "title": "Employment Act 2001",
         "url": "https://example.com/emp", "section_index": 0},
        {"page_content": "Notice period is 4 weeks.", "title": "Employment Act 2001",
         "url": "https://example.com/emp", "section_index": 1},
    ]
    agent = make_researcher_with_vector_store(mock_store)
    result = agent.run("employment notice period")
    assert isinstance(result, ResearcherOutput)
    assert len(result.acts) == 1
    assert result.acts[0].title == "Employment Act 2001"
    assert result.acts[0].url == "https://example.com/emp"
    assert result.acts[0].sections[0] == "Workers must receive notice."
    assert result.acts[0].sections[1] == "Notice period is 4 weeks."


def test_researcher_sections_ordered_by_section_index():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    # Return results out of order
    mock_store.search.return_value = [
        {"page_content": "Second section.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 1},
        {"page_content": "First section.", "title": "Act A",
         "url": "https://example.com/a", "section_index": 0},
    ]
    agent = make_researcher_with_vector_store(mock_store)
    result = agent.run("some query")
    assert result.acts[0].sections[0] == "First section."
    assert result.acts[0].sections[1] == "Second section."


def test_researcher_raises_on_empty_vector_results():
    mock_store = MagicMock()
    mock_store.is_populated.return_value = True
    mock_store.search.return_value = []
    agent = make_researcher_with_vector_store(mock_store)
    import pytest
    with pytest.raises(ValueError, match="No Acts found"):
        agent.run("unknown topic xyz")


def test_researcher_falls_back_when_store_not_populated(caplog):
    mock_store = MagicMock()
    mock_store.is_populated.return_value = False
    agent = make_researcher_with_vector_store(
        mock_store,
        search_results=[{"title": "Act A", "url": "https://example.com/a"}],
        sections_per_act=["Live section text."],
    )
    with caplog.at_level(logging.WARNING, logger="irish_statute_assistant.agents.researcher"):
        result = agent.run("employment law")
    assert isinstance(result, ResearcherOutput)
    assert result.acts[0].sections[0] == "Live section text."
    assert any("vector store" in msg.lower() or "populated" in msg.lower()
               for msg in caplog.messages)
