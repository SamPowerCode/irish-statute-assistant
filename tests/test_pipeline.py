from unittest.mock import MagicMock
from irish_statute_assistant.pipeline import Pipeline
from irish_statute_assistant.models.schemas import WriterOutput, DetailedBreakdown


def make_pipeline(supervisor_returns):
    from unittest.mock import MagicMock
    from irish_statute_assistant.memory.session_memory import SessionMemory
    p = Pipeline.__new__(Pipeline)
    mock_config = MagicMock()
    mock_config.token_budget_per_query = 4000
    p._config = mock_config
    p._supervisor = MagicMock()
    p._supervisor.run = MagicMock(return_value=supervisor_returns)
    p._memory = SessionMemory()
    return p


def writer_output():
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=["6 year limit"], caveats=["Seek advice"],
        )
    )


def test_pipeline_returns_writer_output():
    p = make_pipeline(writer_output())
    result = p.query("How long do I have?")
    assert isinstance(result, WriterOutput)


def test_pipeline_returns_clarification_string():
    p = make_pipeline("Can you be more specific?")
    result = p.query("What are my rights?")
    assert isinstance(result, str)
    assert result == "Can you be more specific?"


def test_pipeline_memory_updated_after_answer():
    p = make_pipeline(writer_output())
    p.query("How long do I have?")
    history = p._memory.get_history()
    assert len(history) == 1
    assert history[0]["user"] == "How long do I have?"


def test_pipeline_memory_not_updated_after_clarification():
    p = make_pipeline("Can you be more specific?")
    p.query("What are my rights?")
    history = p._memory.get_history()
    assert len(history) == 0
