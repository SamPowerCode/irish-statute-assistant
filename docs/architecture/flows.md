# Detailed Flows

Sequence diagrams for the three key flows in the pipeline.

## Clarification flow

When a query is ambiguous, the clarifier asks one focused question. The exchange
is stored in memory so the follow-up answer has context.

```{mermaid}
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant Cl as Clarifier
    participant M as ConversationStore

    U->>P: query("What are my rights?")
    P->>Cl: run(query, history="")
    Cl-->>P: ClarifierOutput(needs_clarification=True, question="What area of law?")
    P->>M: add_exchange(user=query, assistant=question)
    P-->>U: "What area of law?"

    U->>P: query("Employment rights")
    P->>Cl: run("Employment rights", history)
    Cl-->>P: ClarifierOutput(needs_clarification=False)
    Note over P: Pipeline continues to Researcher
```

## Refinement loop

The writer, grounding checker, and evaluator iterate until quality passes or
retries are exhausted. Each retry carries the evaluator's flags and the latest
devil's advocate challenges into the next writer call.

```{mermaid}
sequenceDiagram
    participant S as Supervisor
    participant W as Writer
    participant G as GroundingChecker
    participant E as Evaluator
    participant Ad as DevilsAdvocate

    loop Up to effective_refinements + 1 times
        S->>W: run(query, analyst_output, research, evaluator_flags)
        W-->>S: WriterOutput
        S->>G: run(writer_output, research)
        G-->>S: GroundingOutput(ungrounded_claims, grounding_passed)
        S->>S: writer_result.warnings = grounding.ungrounded_claims
        S->>E: run(query, writer_result, grounding_passed)
        E-->>S: EvaluatorOutput(score, flags, pass_)

        alt pass_ is True
            S->>S: memory.add_exchange()
            S-->>S: return WriterOutput
        else retries remain
            S->>S: evaluator_flags = flags
            S->>Ad: run(analyst_output, query, research, mode=advocate_mode)
            Ad-->>S: AdvocateOutput(challenges, severity)
            S->>S: analyst_output.advocate_challenges = challenges
        end
    end
    Note over S: Exhausted — return best attempt
```

## Confidence gate

The confidence gate runs after the analyst and the first devil's advocate call.
It determines how many refinement rounds to allow and whether the devil's
advocate uses standard or strict mode on retries.

```{mermaid}
sequenceDiagram
    participant S as Supervisor
    participant An as Analyst
    participant Ad as DevilsAdvocate

    S->>An: run(query, research)
    An-->>S: AnalystLLMOutput(key_clauses, gaps, confidence)
    S->>S: analyst_output = AnalystOutput(**result, advocate_challenges=[])

    S->>Ad: run(analyst_output, query, research, mode="standard")
    Ad-->>S: AdvocateOutput(challenges, severity)

    alt confidence < 0.5 OR severity == "major"
        Note over S: effective_refinements = max_refinements * 2
        Note over S: advocate_mode_on_retry = "strict"
    else
        Note over S: effective_refinements = max_refinements
        Note over S: advocate_mode_on_retry = "standard"
    end

    S->>S: analyst_output.advocate_challenges = challenges
    Note over S: Enters refinement loop
```
