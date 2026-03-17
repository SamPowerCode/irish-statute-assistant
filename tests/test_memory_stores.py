import os
import tempfile
import pytest
from irish_statute_assistant.memory.conversation_store import ConversationStore


def test_conversation_store_add_and_retrieve(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db, history_limit=20)
    store.add_exchange(user="Hello", assistant="Hi there")
    history = store.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "Hello"
    assert history[0]["assistant"] == "Hi there"


def test_conversation_store_persists_across_instantiations(tmp_path):
    db = str(tmp_path / "conv.db")
    store1 = ConversationStore(db_path=db, history_limit=20)
    store1.add_exchange(user="Q1", assistant="A1")

    store2 = ConversationStore(db_path=db, history_limit=20)
    history = store2.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "Q1"


def test_conversation_store_history_limit_enforced(tmp_path):
    db = str(tmp_path / "conv.db")
    # Write 5 exchanges to same DB, then load with limit=3
    store_write = ConversationStore(db_path=db, history_limit=10)
    for i in range(5):
        store_write.add_exchange(user=f"Q{i}", assistant=f"A{i}")

    store_read = ConversationStore(db_path=db, history_limit=3)
    history = store_read.get_history()
    assert len(history) == 3
    # Should be the 3 most recent
    assert history[0]["user"] == "Q2"
    assert history[2]["user"] == "Q4"


def test_conversation_store_format_for_prompt(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db)
    store.add_exchange(user="What are my rights?", assistant="That depends on context.")
    prompt = store.format_for_prompt()
    assert "User: What are my rights?" in prompt
    assert "Assistant: That depends on context." in prompt


def test_conversation_store_empty_format_for_prompt(tmp_path):
    db = str(tmp_path / "conv.db")
    store = ConversationStore(db_path=db)
    assert store.format_for_prompt() == ""


def test_conversation_store_creates_db_dir_on_first_use(tmp_path):
    nested = str(tmp_path / "a" / "b" / "conv.db")
    store = ConversationStore(db_path=nested)
    store.add_exchange(user="x", assistant="y")
    assert os.path.exists(nested)


from irish_statute_assistant.memory.user_preference_store import UserPreferenceStore


def test_preference_store_set_and_get(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("language_level", "plain")
    assert store.get("language_level") == "plain"


def test_preference_store_default_when_missing(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    assert store.get("nonexistent", default="fallback") == "fallback"


def test_preference_store_overwrite(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("verbosity", "brief")
    store.set("verbosity", "detailed")
    assert store.get("verbosity") == "detailed"


def test_preference_store_all(tmp_path):
    db = str(tmp_path / "prefs.db")
    store = UserPreferenceStore(db_path=db)
    store.set("user_type", "solicitor")
    store.set("language_level", "technical")
    prefs = store.all()
    assert prefs == {"user_type": "solicitor", "language_level": "technical"}


def test_preference_store_persists_across_instantiations(tmp_path):
    db = str(tmp_path / "prefs.db")
    UserPreferenceStore(db_path=db).set("user_type", "solicitor")
    assert UserPreferenceStore(db_path=db).get("user_type") == "solicitor"
