# Testing

## Running the tests

```bash
# All tests
python -m pytest

# With verbose output
python -m pytest -v

# A specific file
python -m pytest tests/test_supervisor.py -v

# A specific test
python -m pytest tests/test_supervisor.py::test_supervisor_returns_writer_output_when_clear_and_passes -v
```

Expected: All tests pass (no failures).

## Test file overview

| File | What it covers |
|---|---|
| `test_schemas.py` | Pydantic validation: required fields, constraints, aliases |
| `test_analyst.py` | Analyst returns `AnalystLLMOutput`; no `evaluator_flags` param |
| `test_writer.py` | `KeyClause` serialisation; `advocate_challenges` injection |
| `test_evaluator.py` | Citation serialisation; `grounding_passed` flag effect |
| `test_devils_advocate.py` | Chain routing by mode; challenge/severity behaviour |
| `test_grounding_checker.py` | Grounded vs ungrounded claims; input formatting |
| `test_supervisor.py` | Full pipeline flow; confidence gate; memory writes; preferences |
| `test_pipeline.py` | Pipeline wiring; QueryContext forwarding |
| `test_memory_stores.py` | SQLite persistence; history limits; preference read/write |
| `test_config.py` | Config loading from env; provider key validation |
| `test_adversarial.py` | Error propagation; budget exceeded; statute not found |

## Testing patterns

### Bypass `__init__` with `__new__`

Agent tests bypass the constructor (which needs a real `Config` and LLM) using
`__new__`, then inject mock attributes directly:

```python
agent = WriterAgent.__new__(WriterAgent)
mock_chain = MagicMock()
mock_chain.invoke = MagicMock(return_value=WriterOutput(...))
agent._chain = mock_chain
```

### Mock `_chain.invoke`, not `_invoke_chain`

Most agents store the LangChain chain as `self._chain`. Tests mock `_chain.invoke`
directly so the agent's `run()` method is exercised end-to-end through
`_invoke_chain`, while the actual LLM call is intercepted at the chain level:

```python
mock_chain = MagicMock()
mock_chain.invoke = MagicMock(return_value=SomeOutput(...))
agent._chain = mock_chain
result = agent.run(...)
mock_chain.invoke.assert_called_once()
```

Some agents (e.g. `GroundingCheckerAgent`) mock `_invoke_chain` directly instead,
which is appropriate when you want to verify the exact inputs passed to that method:

```python
agent._invoke_chain = MagicMock(return_value=SomeOutput(...))
result = agent.run(...)
agent._invoke_chain.assert_called_once()
```

### SQLite fixtures with `tmp_path`

Memory store tests use pytest's `tmp_path` fixture for isolated temporary databases:

```python
def test_conversation_store_persists(tmp_path):
    db = str(tmp_path / "conv.db")
    store1 = ConversationStore(db_path=db)
    store1.add_exchange(user="Q", assistant="A")
    store2 = ConversationStore(db_path=db)
    assert store2.get_history()[0]["user"] == "Q"
```

### Supervisor tests use `Supervisor.__new__`

The Supervisor test helper creates a fully-wired mock supervisor without
touching any infrastructure:

```python
sup = Supervisor.__new__(Supervisor)
sup._max_refinements = 2
sup._max_retries = 3
sup._evaluator_flag_counts = {}
sup._memory = MagicMock()
sup._clarifier = MagicMock()
# ... etc
```
