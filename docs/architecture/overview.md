# Architecture Overview

The Irish Statute Assistant uses a multi-agent pipeline. Each agent has a single
responsibility and communicates through typed Pydantic schemas. The pipeline runs
sequentially — there is no parallelism.

## Pipeline

```{mermaid}
flowchart TD
    A([User Query]) --> B[Clarifier]
    B -->|needs clarification| C([Clarifying Question → User])
    B -->|clear| D[Researcher]
    D --> E[Analyst]
    E --> F[Devil's Advocate]
    F --> G{Confidence Gate}
    G -->|low confidence or major challenge| H["Refinement Loop\n(doubled rounds, strict mode)"]
    G -->|normal| I["Refinement Loop\n(standard rounds)"]
    H --> J[Writer]
    I --> J
    J --> K[Grounding Checker]
    K --> L[Evaluator]
    L -->|pass| M([Answer → User])
    L -->|fail, retries remain| J
    L -->|fail, exhausted| M
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
