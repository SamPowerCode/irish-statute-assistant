import json
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
def sample_solr_search_response():
    """Minimal Solr JSON response mimicking irishstatutebook.ie title search.

    The live site uses the Solr endpoint /solr/all_leg_title/select which
    returns JSON.  Results are filtered client-side to type == 'act'.
    """
    return json.dumps({
        "response": {
            "docs": [
                {
                    "title": "Civil Liability and Courts Act 2004",
                    "link": "https://www.irishstatutebook.ie/2004/en/act/pub/0024/index.html",
                    "type": "act",
                    "year": "2004",
                    "act": "24",
                }
            ]
        }
    })


@pytest.fixture
def sample_html_act_page():
    """Minimal HTML mimicking an Act page on irishstatutebook.ie.

    The live site uses table-based layout.  Sections are contained in
    <table class="t1"> elements inside <div id="act" class="act-content">.
    There are no <div class="section"> elements in the real HTML.
    """
    return """
    <html><body>
      <div class="act-content" id="act">
        <table class="t1" width="100%">
          <tr>
            <td valign="top"><p>1.</p></td>
            <td valign="top">
              <p>Short title and commencement.</p>
              <p>A person who suffers a personal injury shall bring a claim within two years.</p>
            </td>
          </tr>
        </table>
        <table class="t1" width="100%">
          <tr>
            <td valign="top"><p>2.</p></td>
            <td valign="top">
              <p>Interpretation.</p>
              <p>The court may extend this period in exceptional circumstances.</p>
            </td>
          </tr>
        </table>
      </div>
    </body></html>
    """
