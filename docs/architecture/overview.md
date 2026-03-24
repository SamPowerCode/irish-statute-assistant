# Architecture Overview

The Irish Statute Assistant uses a multi-agent pipeline. Each agent has a single
responsibility and communicates through typed Pydantic schemas. The pipeline runs
sequentially — there is no parallelism.

## Pipeline

```{mermaid}
flowchart TD
    Q([User Query]) --> Clarifier
    ConvStore[(Conversation History)] -->|prior exchanges| Clarifier

    Clarifier -->|ambiguous| CQ([Clarifying question to user])
    Clarifier -->|clear| Researcher

    VS[(Vector Store - ChromaDB / Qdrant)] --> Researcher
    ISB[irishstatutebook.ie] -. live fallback .-> Researcher

    Researcher --> Analyst
    Analyst --> DA["Devil's Advocate"]
    DA --> Writer

    PrefStore[(User Preferences)] -->|language & verbosity| Writer

    Writer --> GC[Grounding Checker]
    GC --> Eval[Evaluator]

    Eval -->|"score >= 0.7 - pass"| Answer([Answer to User])
    Eval -->|"score < 0.7 - retry"| DA

    Answer --> ConvStore
    Answer --> PrefStore
```

## Agents at a glance

| Agent | Runs | Purpose |
|---|---|---|
| **Clarifier** | Once, first | Asks one question if the query is ambiguous |
| **Researcher** | Once | Retrieves relevant Acts from the vector store |
| **Analyst** | Once | Identifies key clauses and scores confidence |
| **Devil's Advocate** | Once + on each retry | Challenges the analyst's conclusions |
| **Writer** | Each loop iteration | Produces the plain-English answer |
| **Grounding Checker** | Each loop iteration | Verifies citations against retrieved text |
| **Evaluator** | Each loop iteration | Scores quality and decides whether to retry |
| **Supervisor** | Orchestrator | Runs all agents, owns memory writes, detects preferences |

## Refinement loop

The writer, grounding checker, and evaluator form a refinement loop. If the
evaluator's score falls below `EVALUATOR_PASS_THRESHOLD` (default 0.7), the
writer runs again with the evaluator's flags as feedback. This repeats up to
`MAX_REFINEMENT_ROUNDS` times, after which the best attempt is returned.

## Confidence gate

After the analyst runs, the devil's advocate evaluates its findings. If the
analyst's confidence is below 0.5, or if the devil's advocate rates the
problem as "major", the refinement rounds are doubled and the devil's advocate
switches to strict mode on retries. See [Flows](flows.md) for a sequence diagram.

## Memory

Two SQLite stores persist state across sessions:

- **ConversationStore** — stores every user/assistant exchange; injected into
  the clarifier's prompt so the assistant maintains context across questions
- **UserPreferenceStore** — stores detected preferences (language level,
  verbosity, user type); injected into the analyst and writer prompts

See [Memory](memory.md) for details.
