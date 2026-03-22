# Running the Assistant

## Start

```bash
uv run python -m irish_statute_assistant.main
```

You will see a prompt:

```
Irish Statute Research Assistant
Type your legal question, or 'quit' to exit.

Your question:
```

## Streamlit UI

If you installed the `ui` extra (see [Installation](installation.md#optional-streamlit-ui)):

```bash
uv run streamlit run app.py
```

Opens the assistant at `http://localhost:8501`. The interface has two panels:

- **Main area** — chat conversation: type your question, read the response
- **Sidebar** — live pipeline trace: shows each agent as it runs, with timing and key stats (acts found, confidence score, evaluator result, etc.)

---

## Example interaction

```
Your question: How long do I have to bring a personal injury claim?

Answer: In Ireland you generally have two years from the date of the injury
to bring a personal injury claim, though this period may be extended in
certain circumstances such as when the injured person is a minor.

--- Detail ---
Summary: The Statute of Limitations Act 1957, as amended by the Civil
Liability and Courts Act 2004, sets a two-year limitation period for
personal injury claims.

Relevant Acts:
  - Statute of Limitations Act 1957
  - Civil Liability and Courts Act 2004

Key points:
  - Two-year limitation period for personal injury. (Statute of Limitations
    Act 1957, s.11)
  - Period runs from date of injury or date of knowledge. (Civil Liability
    and Courts Act 2004, s.7)

Things to be aware of:
  - The limitation period may be extended if the claimant was a minor at the
    time of injury.
  - Seek professional legal advice — this summary does not constitute legal
    advice.
```

### Parts of the output

| Section | What it means |
|---|---|
| **Answer** | A plain-English summary in 100 words or fewer |
| **Summary** | A 2–3 sentence overview of the legal position |
| **Relevant Acts** | The Irish Acts that apply to the question |
| **Key points** | Specific rules, each citing the Act name and section |
| **Things to be aware of** | Exceptions, edge cases, and reminders to seek advice |

### Additional notices

If the assistant is uncertain about statute coverage, you may see:

```
Note: confidence in statute coverage was low for this query.
```

If any key points could not be verified against retrieved statute text:

```
--- Grounding warnings ---
  - [description of unverified claim]
```

## Clarifying questions

If your question is ambiguous, the assistant will ask one clarifying question
before proceeding:

```
Your question: What are my rights?

I need a bit more information:
  What area of law are you asking about — for example, employment,
  housing, or consumer rights?

Your clarification: Employment rights
```

The assistant remembers the conversation within a session, so follow-up
questions are understood in context.

## Ending a session

Type `quit`, `exit`, or `q` (or press `Ctrl+C`) to exit. Conversation history is saved
automatically to `~/.irish_statute_assistant/conversations.db`.
