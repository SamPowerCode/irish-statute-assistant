from unittest.mock import MagicMock, patch
from irish_statute_assistant.agents.researcher import ResearcherAgent
from irish_statute_assistant.models.schemas import ResearcherOutput, ActSection
from irish_statute_assistant.tools.session_cache import SessionCache


def make_researcher_with_mocks(search_results, sections_per_act):
    cache = SessionCache()
    agent = ResearcherAgent.__new__(ResearcherAgent)
    agent._cache = cache

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
