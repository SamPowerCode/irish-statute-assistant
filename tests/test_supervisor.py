import pytest
from unittest.mock import MagicMock
from irish_statute_assistant.agents.supervisor import Supervisor
from irish_statute_assistant.models.schemas import (
    ClarifierOutput, ResearcherOutput, ActSection,
    AnalystLLMOutput, AnalystOutput, KeyClause,
    WriterOutput, DetailedBreakdown,
    EvaluatorOutput, AdvocateOutput, GroundingOutput,
)


def make_key_clause():
    return KeyClause(text="6 year limit", act="Act A", section="s.1")


def make_defaults():
    research = ResearcherOutput(acts=[
        ActSection(title="Act A", url="https://example.com", sections=["Section 1"])
    ])
    kc = make_key_clause()
    analyst_llm = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.9)
    writer_out = WriterOutput(
        short_answer="You have six years.",
        detailed_breakdown=DetailedBreakdown(
            summary="Six year limit.", relevant_acts=["Act A"],
            key_clauses=[kc], caveats=["Seek advice"],
        )
    )
    evaluator_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})
    advocate_minor = AdvocateOutput(challenges=[], severity="minor")
    grounding_pass = GroundingOutput(ungrounded_claims=[], grounding_passed=True)
    return research, analyst_llm, writer_out, evaluator_pass, advocate_minor, grounding_pass


def make_supervisor(
    clarifier_output, researcher_output, analyst_llm_output,
    writer_output, evaluator_output, advocate_output, grounding_output,
):
    sup = Supervisor.__new__(Supervisor)
    sup._max_refinements = 2
    sup._max_retries = 3
    sup._evaluator_flag_counts = {}

    sup._memory = MagicMock()
    sup._memory.format_for_prompt = MagicMock(return_value="")
    sup._memory.add_exchange = MagicMock()

    sup._preferences = MagicMock()
    sup._preferences.set = MagicMock()
    sup._preferences.get = MagicMock(return_value="")
    sup._preferences.all = MagicMock(return_value={})

    sup._clarifier = MagicMock()
    sup._clarifier.run = MagicMock(return_value=clarifier_output)
    sup._clarifier.last_token_count = 0

    sup._researcher = MagicMock()
    sup._researcher.run = MagicMock(return_value=researcher_output)
    sup._researcher.last_token_count = 0

    sup._analyst = MagicMock()
    sup._analyst.run = MagicMock(return_value=analyst_llm_output)
    sup._analyst.last_token_count = 0

    sup._advocate = MagicMock()
    sup._advocate.run = MagicMock(return_value=advocate_output)
    sup._advocate.last_token_count = 0

    sup._writer = MagicMock()
    sup._writer.run = MagicMock(return_value=writer_output)
    sup._writer.last_token_count = 0

    sup._grounding_checker = MagicMock()
    sup._grounding_checker.run = MagicMock(return_value=grounding_output)
    sup._grounding_checker.last_token_count = 0

    sup._evaluator = MagicMock()
    sup._evaluator.run = MagicMock(return_value=evaluator_output)
    sup._evaluator.last_token_count = 0

    return sup


def test_supervisor_returns_writer_output_when_clear_and_passes():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    result = sup.run(query="How long do I have to sue?", context=None)
    assert isinstance(result, WriterOutput)


def test_supervisor_returns_clarification_question_when_needed():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=True, question="What type of case?"),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    result = sup.run(query="What are my rights?", context=None)
    assert result == "What type of case?"
    sup._memory.add_exchange.assert_called_once_with(
        user="What are my rights?", assistant="What type of case?"
    )


def test_supervisor_refinement_loop_retries_on_fail():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_fail = EvaluatorOutput(score=0.5, flags=["Vague answer"], **{"pass": False})
    eval_pass = EvaluatorOutput(score=0.85, flags=[], **{"pass": True})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_fail,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup._evaluator.run = MagicMock(side_effect=[eval_fail, eval_pass])
    result = sup.run(query="How long?", context=None)
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == 2


def test_supervisor_stops_after_max_refinements():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_fail = EvaluatorOutput(score=0.4, flags=["Still bad"], **{"pass": False})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_fail,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    result = sup.run(query="How long?", context=None)
    assert isinstance(result, WriterOutput)
    assert sup._evaluator.run.call_count == sup._max_refinements + 1


def test_supervisor_confidence_gate_doubles_refinements_on_low_confidence():
    research, _, writer_out, eval_pass, _, grounding = make_defaults()
    kc = make_key_clause()
    low_conf_analyst = AnalystLLMOutput(key_clauses=[kc], gaps=[], confidence=0.3)
    advocate_minor = AdvocateOutput(challenges=[], severity="minor")

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=low_conf_analyst,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate_minor,
        grounding_output=grounding,
    )
    # Capture effective_refinements by making evaluator always fail until exhausted
    eval_fail = EvaluatorOutput(score=0.4, flags=["bad"], **{"pass": False})
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    sup.run(query="Q", context=None)
    # With low confidence, should run max_refinements*2 + 1 times
    assert sup._evaluator.run.call_count == sup._max_refinements * 2 + 1


def test_supervisor_confidence_gate_doubles_refinements_on_major_severity():
    research, analyst_llm, writer_out, eval_pass, _, grounding = make_defaults()
    advocate_major = AdvocateOutput(challenges=["Serious problem"], severity="major")

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate_major,
        grounding_output=grounding,
    )
    eval_fail = EvaluatorOutput(score=0.4, flags=["bad"], **{"pass": False})
    sup._evaluator.run = MagicMock(return_value=eval_fail)
    sup.run(query="Q", context=None)
    assert sup._evaluator.run.call_count == sup._max_refinements * 2 + 1


def test_supervisor_memory_add_exchange_called_on_success():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup.run(query="Q", context=None)
    sup._memory.add_exchange.assert_called_once_with(
        user="Q", assistant=writer_out.short_answer
    )


def test_supervisor_explicit_preference_saved_on_solicitor_query():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup.run(query="I'm a solicitor — what are the rules on X?", context=None)
    sup._preferences.set.assert_any_call("user_type", "solicitor")


def test_supervisor_passes_preferences_to_writer():
    research, analyst_llm, writer_out, eval_pass, advocate, grounding = make_defaults()
    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_pass,
        advocate_output=advocate,
        grounding_output=grounding,
    )
    sup._preferences.all = MagicMock(return_value={"language_level": "technical"})
    sup.run(query="Q", context=None)
    call_kwargs = sup._writer.run.call_args[1]
    assert call_kwargs.get("user_preferences") == {"language_level": "technical"}


def test_supervisor_inferred_preference_saved_after_second_plain_english_flag():
    research, analyst_llm, writer_out, _, advocate, grounding = make_defaults()
    eval_with_flag = EvaluatorOutput(score=0.75, flags=["plain english"], **{"pass": True})

    sup = make_supervisor(
        clarifier_output=ClarifierOutput(needs_clarification=False),
        researcher_output=research,
        analyst_llm_output=analyst_llm,
        writer_output=writer_out,
        evaluator_output=eval_with_flag,
        advocate_output=advocate,
        grounding_output=grounding,
    )

    # First query — should NOT save preference
    sup.run(query="Q1", context=None)
    calls_after_first = [c for c in sup._preferences.set.call_args_list
                         if c[0] == ("language_level", "technical")]
    assert len(calls_after_first) == 0

    # Reset evaluator mock for second query
    sup._evaluator.run = MagicMock(return_value=eval_with_flag)
    sup._analyst.run = MagicMock(return_value=analyst_llm)
    sup._advocate.run = MagicMock(return_value=advocate)
    sup._grounding_checker.run = MagicMock(return_value=grounding)
    sup._writer.run = MagicMock(return_value=writer_out)
    sup._researcher.run = MagicMock(return_value=research)
    sup._clarifier.run = MagicMock(return_value=ClarifierOutput(needs_clarification=False))

    # Second query — should save preference
    sup.run(query="Q2", context=None)
    calls_after_second = [c for c in sup._preferences.set.call_args_list
                          if c[0] == ("language_level", "technical")]
    assert len(calls_after_second) == 1
