from unittest.mock import MagicMock, patch
from irish_statute_assistant.pipeline import Pipeline
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.models.schemas import (
    WriterOutput, DetailedBreakdown, KeyClause
)
from irish_statute_assistant.config import Config


def make_writer_output():
    kc = KeyClause(text="6 year limit", act="Act A", section="s.1")
    return WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=[kc], caveats=["Seek advice"],
        )
    )


def make_pipeline(writer_output=None, clarification=None):
    """Create a Pipeline with mocked supervisor."""
    return_value = clarification if clarification else (writer_output or make_writer_output())
    with patch("irish_statute_assistant.pipeline.ConversationStore"), \
         patch("irish_statute_assistant.pipeline.UserPreferenceStore"), \
         patch("irish_statute_assistant.pipeline.Supervisor") as mock_sup_class:
        mock_sup = MagicMock()
        mock_sup.run = MagicMock(return_value=return_value)
        mock_sup_class.return_value = mock_sup
        config = MagicMock(spec=Config)
        config.token_budget_per_query = 20000
        config.conversations_db_path = ":memory:"
        config.conversation_history_limit = 20
        config.preferences_db_path = ":memory:"
        pipeline = Pipeline(config)
    return pipeline, mock_sup


def test_pipeline_query_returns_writer_output():
    writer_out = make_writer_output()
    pipeline, mock_sup = make_pipeline(writer_output=writer_out)
    result = pipeline.query("How long do I have to sue?")
    assert isinstance(result, WriterOutput)
    mock_sup.run.assert_called_once()


def test_pipeline_query_returns_clarification_string():
    pipeline, mock_sup = make_pipeline(clarification="Can you clarify your question?")
    result = pipeline.query("What are my rights?")
    assert result == "Can you clarify your question?"


def test_pipeline_query_passes_context_to_supervisor():
    from irish_statute_assistant.context import QueryContext
    pipeline, mock_sup = make_pipeline()
    pipeline.query("Q")
    call_kwargs = mock_sup.run.call_args
    context_arg = mock_sup.run.call_args.kwargs["context"]
    assert isinstance(context_arg, QueryContext)
