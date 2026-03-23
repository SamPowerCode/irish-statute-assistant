import pytest
import httpx
from pytest_httpx import HTTPXMock
from irish_statute_assistant.tools.statute_fetcher import StatuteFetcher
from irish_statute_assistant.tools.session_cache import SessionCache


SOLR_URL = "https://www.irishstatutebook.ie/solr/all_leg_title/select"
ACT_URL = "https://www.irishstatutebook.ie/eli/2004/act/24/enacted/en/html"


@pytest.fixture
def fetcher():
    return StatuteFetcher(rate_limit_delay=0.0, max_retries=3)


def test_search_statutes_returns_results(fetcher, httpx_mock: HTTPXMock, sample_solr_search_response):
    httpx_mock.add_response(
        url=f"{SOLR_URL}?q=personal+injury&wt=json&rows=15",
        text=sample_solr_search_response,
    )
    results = fetcher.search("personal injury")
    assert len(results) >= 1
    assert "title" in results[0]
    assert "url" in results[0]


def test_search_statutes_returns_only_acts(fetcher, httpx_mock: HTTPXMock):
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
        url=f"{SOLR_URL}?q=civil&wt=json&rows=15",
        text=mixed_response,
    )
    results = fetcher.search("civil")
    assert len(results) == 1
    assert results[0]["title"] == "Civil Liability Act 2004"


def test_search_statutes_returns_at_most_5(fetcher, httpx_mock: HTTPXMock):
    """Even if Solr returns more than 5 acts, we cap at 5."""
    import json
    docs = [
        {"title": f"Act {i}", "link": f"https://www.irishstatutebook.ie/200{i}/en/act/pub/000{i}/index.html", "type": "act", "year": f"200{i}"}
        for i in range(10)
    ]
    response = json.dumps({"response": {"docs": docs}})
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json&rows=15", text=response)
    results = fetcher.search("test")
    assert len(results) <= 5


def test_fetch_act_sections_returns_sections(fetcher, httpx_mock: HTTPXMock, sample_html_act_page):
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    sections = fetcher.fetch(ACT_URL, cache)
    assert len(sections) >= 1
    assert isinstance(sections[0], str)


def test_fetch_act_sections_uses_cache(fetcher, httpx_mock: HTTPXMock, sample_html_act_page):
    """Second call should not make an HTTP request."""
    httpx_mock.add_response(url=ACT_URL, text=sample_html_act_page)
    cache = SessionCache()
    fetcher.fetch(ACT_URL, cache)      # first call — hits network
    fetcher.fetch(ACT_URL, cache)      # second call — from cache
    assert len(httpx_mock.get_requests()) == 1


def test_search_statutes_retries_on_failure(fetcher, httpx_mock: HTTPXMock, sample_solr_search_response):
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json&rows=15", status_code=500)
    httpx_mock.add_response(url=f"{SOLR_URL}?q=test&wt=json&rows=15", text=sample_solr_search_response)
    results = fetcher.search("test")
    assert len(results) >= 1
