# Adding an Agent

This walkthrough adds a hypothetical `SummarizerAgent` that produces a one-sentence
summary of the writer's output. Use it as a template for any new agent.

## 1. Define an output schema

Add your schema to `src/irish_statute_assistant/models/schemas.py`:

```python
class SummarizerOutput(BaseModel):
    """One-sentence summary produced by the SummarizerAgent."""
    one_liner: str
```

## 2. Create the agent file

Create `src/irish_statute_assistant/agents/summarizer.py`:

```python
from langchain_core.prompts import ChatPromptTemplate

from irish_statute_assistant.agents.base_agent import BaseAgent
from irish_statute_assistant.config import Config
from irish_statute_assistant.llm import get_llm
from irish_statute_assistant.models.schemas import SummarizerOutput, WriterOutput

SYSTEM_PROMPT = "You produce one-sentence summaries of legal answers."
HUMAN_PROMPT = "Summarise this answer in one sentence: {short_answer}"


class SummarizerAgent(BaseAgent):
    """Produces a one-sentence summary of the writer's output.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        llm = get_llm(config, max_tokens=128).with_structured_output(SummarizerOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = prompt | llm

    def run(self, writer_output: WriterOutput) -> SummarizerOutput:
        """Summarise the writer's short answer.

        Args:
            writer_output: The writer's answer.

        Returns:
            SummarizerOutput with a one-liner summary.
        """
        return self._invoke_chain(self._chain, {
            "short_answer": writer_output.short_answer,
        })
```

Always use `self._invoke_chain(chain, inputs)` — never `chain.invoke(inputs)` directly.
This ensures token usage is tracked via `BaseAgent.last_token_count`.

## 3. Wire into the Supervisor

In `src/irish_statute_assistant/agents/supervisor.py`:

```python
from irish_statute_assistant.agents.summarizer import SummarizerAgent

class Supervisor:
    def __init__(self, config, memory, preferences):
        ...
        self._summarizer = SummarizerAgent(config)

    def run(self, query, context=None):
        ...
        # After the writer runs:
        summary = run_with_retry(
            lambda: self._summarizer.run(writer_result),
            self._max_retries,
        )
        if context:
            context.consume(self._summarizer.last_token_count)
        # Note: one_liner must be added as a field to WriterOutput in schemas.py first
        writer_result.one_liner = summary.one_liner
```

## 4. Write tests

Follow the project pattern — bypass `__init__` using `__new__` and mock `_chain.invoke`:

```python
from unittest.mock import MagicMock
from irish_statute_assistant.agents.summarizer import SummarizerAgent
from irish_statute_assistant.models.schemas import SummarizerOutput, WriterOutput, DetailedBreakdown, KeyClause

def make_summarizer(one_liner):
    agent = SummarizerAgent.__new__(SummarizerAgent)
    mock_chain = MagicMock()
    mock_chain.invoke = MagicMock(return_value=SummarizerOutput(one_liner=one_liner))
    agent._chain = mock_chain
    return agent

def test_summarizer_returns_output():
    agent = make_summarizer("You have two years to claim.")
    kc = KeyClause(text="Two year limit", act="Act A", section="s.1")
    writer_out = WriterOutput(
        short_answer="You have two years.",
        detailed_breakdown=DetailedBreakdown(
            summary="S.", relevant_acts=[], key_clauses=[kc], caveats=[]
        )
    )
    result = agent.run(writer_output=writer_out)
    assert result.one_liner == "You have two years to claim."
    agent._chain.invoke.assert_called_once()
```

## When to use `with_structured_output`

Always use `with_structured_output(YourSchema)` on the LLM when your agent
must return structured data. This constrains the model to valid JSON matching
the schema, and enables validation retry via `run_with_retry`.

If you only need free-text output (unusual), use a plain chain without
`with_structured_output` and return a string.
