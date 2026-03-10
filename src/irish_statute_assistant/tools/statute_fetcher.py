"""Statute fetcher for irishstatutebook.ie.

Search uses the site's Solr JSON API (/solr/all_leg_title/select) rather than
scraping the JavaScript-rendered search results page, which never contains
result links in the static HTML.

Act section extraction parses the table-based layout used on enacted Act pages:
sections are wrapped in <table class="t1"> elements inside <div id="act">.
"""

import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from irish_statute_assistant.tools.session_cache import SessionCache

BASE_URL = "https://www.irishstatutebook.ie"

# Solr JSON endpoint for title searches — returns structured JSON directly.
# The JavaScript-rendered HTML search page (/eli/ResultsTitle.html) populates
# results via AJAX after page load, so there is nothing to scrape in the
# static HTML.  Querying Solr directly is simpler and more reliable.
SOLR_SEARCH_URL = f"{BASE_URL}/solr/all_leg_title/select"

# CSS selector for Act sections.  The enacted HTML Act pages use a
# table-based layout; each numbered section sits inside a <table class="t1">
# contained within <div id="act" class="act-content">.  There are no
# <div class="section"> elements in the real site HTML.
SECTION_SELECTOR = "div#act table.t1"

RATE_LIMIT_DELAY = 1.0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def search_statutes(query: str) -> list[dict]:
    """Search irishstatutebook.ie and return up to 5 Act results as {title, url}.

    Uses the Solr JSON API directly.  Results are filtered to type == 'act'
    so that Statutory Instruments are excluded.
    """
    params = {"q": query, "wt": "json"}
    response = httpx.get(SOLR_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    docs = data.get("response", {}).get("docs", [])

    results = []
    for doc in docs:
        if doc.get("type") != "act":
            continue
        link = doc.get("link", "")
        url = link if link.startswith("http") else BASE_URL + link
        results.append({"title": doc.get("title", ""), "url": url})
        if len(results) >= 5:
            break

    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_act_sections(url: str, cache: SessionCache) -> list[str]:
    """Fetch an Act page and return up to 10 section texts. Uses cache."""
    cached = cache.get(url)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_DELAY)
    response = httpx.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    sections = []
    for table in soup.select(SECTION_SELECTOR)[:10]:
        text = table.get_text(strip=True)
        if text:
            sections.append(text[:2000])

    cache.set(url, sections)
    return sections
