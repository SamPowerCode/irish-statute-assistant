from irish_statute_assistant.tools.session_cache import SessionCache


def test_cache_miss_returns_none():
    cache = SessionCache()
    assert cache.get("https://example.com/page") is None


def test_cache_hit_returns_stored_value():
    cache = SessionCache()
    cache.set("https://example.com/page", ["section 1", "section 2"])
    assert cache.get("https://example.com/page") == ["section 1", "section 2"]


def test_cache_is_isolated_per_instance():
    cache_a = SessionCache()
    cache_b = SessionCache()
    cache_a.set("https://example.com/page", ["data"])
    assert cache_b.get("https://example.com/page") is None


def test_cache_overwrite():
    cache = SessionCache()
    cache.set("https://example.com/page", ["old"])
    cache.set("https://example.com/page", ["new"])
    assert cache.get("https://example.com/page") == ["new"]
