import pytest
import httpx
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """A mock LangChain chat model that returns a preset string."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=MagicMock(content="{}"))
    return llm


@pytest.fixture
def sample_html_search_results():
    """Minimal HTML mimicking irishstatutebook.ie search results."""
    return """
    <html><body>
      <ul class="searchresults">
        <li class="result">
          <a href="/eli/2004/act/24/enacted/en/html">
            Civil Liability and Courts Act 2004
          </a>
        </li>
      </ul>
    </body></html>
    """


@pytest.fixture
def sample_html_act_page():
    """Minimal HTML mimicking an Act page on irishstatutebook.ie."""
    return """
    <html><body>
      <div class="section">
        <h3>Section 1</h3>
        <p>A person who suffers a personal injury shall bring a claim within two years.</p>
      </div>
      <div class="section">
        <h3>Section 2</h3>
        <p>The court may extend this period in exceptional circumstances.</p>
      </div>
    </body></html>
    """
