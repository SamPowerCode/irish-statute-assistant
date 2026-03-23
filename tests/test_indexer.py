from unittest.mock import MagicMock, patch

from irish_statute_assistant import indexer as indexer_module


def run_indexer(search_side_effect, fetch_side_effect, acts_per_category=5,
                index_categories=None):
    """Helper: runs indexer.main() with mocked dependencies.

    Returns (sections_passed_to_add_sections, mock_store).
    """
    if index_categories is None:
        index_categories = ["employment", "housing"]

    mock_store = MagicMock()
    captured = {}

    def capture_add(sections):
        captured["sections"] = sections

    mock_store.add_sections.side_effect = capture_add

    mock_config = MagicMock()
    mock_config.acts_per_category = acts_per_category
    mock_config.index_categories = index_categories
    mock_config.rate_limit_delay = 0.0
    mock_config.max_retries = 1
    mock_config.log_level = "INFO"

    mock_fetcher = MagicMock()
    mock_fetcher.search.side_effect = search_side_effect
    mock_fetcher.fetch.side_effect = fetch_side_effect

    with patch("irish_statute_assistant.indexer.StatuteFetcher", return_value=mock_fetcher), \
         patch("irish_statute_assistant.indexer.get_vector_store", return_value=mock_store), \
         patch("irish_statute_assistant.indexer.Config", return_value=mock_config):
        indexer_module.main()

    return captured.get("sections", []), mock_store


def test_indexer_deduplicates_across_categories():
    # URL X appears in both categories — should only be fetched once
    search_results = [
        [{"title": "Act X", "url": "https://example.com/X"},
         {"title": "Act Y", "url": "https://example.com/Y"}],  # employment
        [{"title": "Act X", "url": "https://example.com/X"},
         {"title": "Act Z", "url": "https://example.com/Z"}],  # housing
    ]
    fetch_results = [
        ["Section 1 of X"],  # fetch for X
        ["Section 1 of Y"],  # fetch for Y
        ["Section 1 of Z"],  # fetch for Z
    ]
    sections, mock_store = run_indexer(search_results, fetch_results)
    urls = [s["url"] for s in sections]
    assert urls.count("https://example.com/X") == 1  # fetched once


def test_indexer_respects_acts_per_category():
    # acts_per_category=1: only first unique Act per category
    search_results = [
        [{"title": "Act A", "url": "https://example.com/A"},
         {"title": "Act B", "url": "https://example.com/B"}],  # employment: only A collected
        [{"title": "Act C", "url": "https://example.com/C"},
         {"title": "Act D", "url": "https://example.com/D"}],  # housing: only C collected
    ]
    fetch_results = [
        ["Section of A"],
        ["Section of C"],
    ]
    sections, _ = run_indexer(search_results, fetch_results, acts_per_category=1)
    collected_urls = {s["url"] for s in sections}
    assert "https://example.com/A" in collected_urls
    assert "https://example.com/C" in collected_urls
    assert "https://example.com/B" not in collected_urls
    assert "https://example.com/D" not in collected_urls


def test_indexer_calls_add_sections_once():
    search_results = [
        [{"title": "Act A", "url": "https://example.com/A"}],
        [{"title": "Act B", "url": "https://example.com/B"}],
    ]
    fetch_results = [["S1"], ["S2"]]
    _, mock_store = run_indexer(search_results, fetch_results)
    assert mock_store.add_sections.call_count == 1


def test_indexer_section_dict_shape():
    search_results = [
        [{"title": "Employment Act 2001", "url": "https://example.com/emp"}],
        [],  # housing has no results
    ]
    fetch_results = [["Workers get notice.", "Holiday entitlement applies."]]
    sections, _ = run_indexer(search_results, fetch_results)
    assert len(sections) == 2
    s0 = sections[0]
    assert s0["page_content"] == "Workers get notice."
    assert s0["title"] == "Employment Act 2001"
    assert s0["url"] == "https://example.com/emp"
    assert s0["section_index"] == 0
    assert sections[1]["section_index"] == 1
