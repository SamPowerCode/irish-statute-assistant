# Memory Stores

Two SQLite-backed stores provide persistence across sessions. Both are
constructed by `Pipeline` and passed to the `Supervisor`.

## ConversationStore

Stores every user/assistant exchange. On construction, loads the most recent
`history_limit` exchanges from the database (controlled by
`CONVERSATION_HISTORY_LIMIT` in the environment).

```python
store = ConversationStore(
    db_path="~/.irish_statute_assistant/conversations.db",
    history_limit=20,
)
store.add_exchange(user="How long do I have to sue?", assistant="Six years.")
```

### Integration with agent prompts

`format_for_prompt()` is the bridge between stored history and LLM calls:

```python
history = memory.format_for_prompt()
# Returns:
# "User: How long do I have to sue?
#  Assistant: Six years."
```

The Supervisor passes this string to the Clarifier, which uses it to understand
follow-up questions in context. All writes to `ConversationStore` are owned by
the Supervisor — `Pipeline` never calls `add_exchange()` directly.

### When exchanges are written

- After a clarifying question is returned to the user
- After a successful `WriterOutput` (evaluator passed)
- After the refinement loop is exhausted (best attempt returned)

## UserPreferenceStore

A key-value store for persistent user preferences, detected automatically from
queries and evaluator signals.

```python
store = UserPreferenceStore(db_path="~/.irish_statute_assistant/preferences.db")
store.set("language_level", "plain")
store.get("language_level")   # "plain"
store.all()                   # {"language_level": "plain"}
```

### Detected preferences

**Explicit keyword scan** (case-insensitive, on every query):

| Phrase | Key | Value |
|---|---|---|
| "I'm a solicitor" / "I am a lawyer" | `user_type` | `solicitor` |
| "explain simply" / "plain English" / "non-lawyer" | `language_level` | `plain` |
| "use legal terms" / "technical" | `language_level` | `technical` |
| "brief" / "short answer" | `verbosity` | `brief` |
| "detailed" / "full explanation" | `verbosity` | `detailed` |

**Inferred preference**: If the evaluator returns a `"plain english"` flag on
two or more queries in the same session, the Supervisor saves
`language_level=technical`. The reasoning: a user who repeatedly receives
answers the evaluator flags for failing plain-English criteria is likely
comfortable with legal terminology.

## Why SQLite

Both stores use Python's stdlib `sqlite3` — no extra dependencies, no running
server, works on all platforms, and survives process restarts. The database
directory (`~/.irish_statute_assistant/`) is created automatically on first use.
