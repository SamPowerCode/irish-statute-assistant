import pytest
import httpx
from pytest_httpx import HTTPXMock
from irish_statute_assistant.tools.statute_fetcher import search_statutes, fetch_act_sections
from irish_statute_assistant.tools.session_cache import SessionCache


SOLR_URL = "https://www.irishstatutebook.ie/solr/all_leg_title/select"
ACT_URL = "https://www.irishstatutebook.ie/eli/2004/act/24/enacted/en/html"


def test_search_statutes_returns_results(httpx_mock: HTTPXMock, sample_solr_search_response):
    httpx_mock.add_response(
        url=f"{SOLR_URL}?q=personal+injury&wt=json",
        text=sample_solr_search_response,
    )
    results = search_statutes("personal injury")
    assert len(results) >= 1
    assert "title" in results[0]
    assert "url" in results[0]


def test_search_statutes_returns_only_acts(httpx_mock: HTTPXMock):
    """Results with type != 'act' should be filtered out."""
    import json
    mixed_response = json.dumps({
        "response": {
            "docs": [
                {"title": "Civil Liability Act 2004", "link": "https://www.irishstatutebook.ie/2004/en/act/pub/0024/index.html", "type": "act", "year": "2004"},
                {"title": "S.I. No. 66/1993 - Some SI", "link": "https://www.irishstatutebook.ie/1993/en/si/0066.html", "type": "si", "year": "1993"},
            ]
        }
    })
    httpx_mock.add_response(
        url=f"{SOLR_URL}?q=civil&wt=json",
        text=mixed_response,
    )
    results = search_statutes("civil")
    assert len(results) == 1
    assert results[0]["title"] == "Civil Liability Act 2004"


def test_search_statutes_returns_at_most_5(httpx_mock: HTTPXMock):
    """Even if Solr returns more than 5 acts, we cap at 5."""
    import json
    docs = [
        {"title": f"Act {i}", "link": f"https://www.irishstatutebook.ie/200{i}/en/act/pub/000{i}/index.html", "type": "act", "year": f"200{i}"}
        for i in range(10)
    ]
    response = json.dumps({"response": {"docs": docs}})
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json", text=response)
    results = search_statutes("test")
    assert len(results) <= 5


def test_fetch_act_sections_returns_sections(httpx_mock: HTTPXMock, sample_html_act_page):
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    sections = fetch_act_sections(ACT_URL, cache)
    assert len(sections) >= 1
    assert isinstance(sections[0], str)


def test_fetch_act_sections_uses_cache(httpx_mock: HTTPXMock, sample_html_act_page):
    """Second call should not make an HTTP request."""
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    fetch_act_sections(ACT_URL, cache)      # first call — hits network
    fetch_act_sections(ACT_URL, cache)      # second call — from cache
    assert len(httpx_mock.get_requests()) == 1


def test_search_statutes_retries_on_failure(httpx_mock: HTTPXMock, sample_solr_search_response):
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json", status_code=500)
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json", text=sample_solr_search_response)
    results = search_statutes("test")
    assert len(results) >= 1
