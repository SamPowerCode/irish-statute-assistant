from irish_statute_assistant.memory.session_memory import SessionMemory


def test_empty_memory_returns_empty_history():
    memory = SessionMemory()
    assert memory.get_history() == []


def test_add_exchange_stores_user_and_assistant():
    memory = SessionMemory()
    memory.add_exchange(user="What is the law on X?", assistant="The law says Y.")
    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "What is the law on X?"
    assert history[0]["assistant"] == "The law says Y."


def test_multiple_exchanges_ordered():
    memory = SessionMemory()
    memory.add_exchange(user="Q1", assistant="A1")
    memory.add_exchange(user="Q2", assistant="A2")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["user"] == "Q1"
    assert history[1]["user"] == "Q2"


def test_format_for_prompt_returns_string():
    memory = SessionMemory()
    memory.add_exchange(user="Q1", assistant="A1")
    formatted = memory.format_for_prompt()
    assert "Q1" in formatted
    assert "A1" in formatted
