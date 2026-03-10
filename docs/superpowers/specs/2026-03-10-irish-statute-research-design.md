# Irish Statute Research Assistant — Design Doc

**Date:** 2026-03-10
**Project:** Agentic AI Final Project (7-Week Capstone)
**Level:** Gold
**Framework:** LangChain (Python)
**Knowledge Base:** irishstatutebook.ie

---

## 1. Overview

A multi-agent AI system that answers natural language legal questions from the general public using Irish statute law. The system retrieves relevant Acts from irishstatutebook.ie, interprets them, and returns a plain English answer backed by structured citations. When a query is ambiguous, the system asks one clarifying question before proceeding.

---

## 2. Agents

| Agent | Role |
|-------|------|
| **Supervisor** | Entry point. Parses query, checks session memory, decides if clarification is needed, routes to workers, manages refinement loop. |
| **Clarifier** | Generates a single focused clarifying question when the Supervisor determines the query is ambiguous. |
| **Legal Researcher** | RAG agent. Fetches and retrieves relevant Acts and sections from irishstatutebook.ie based on the query. |
| **Legal Analyst** | Interprets retrieved statutes. Identifies key clauses, flags gaps or conflicts, assigns a confidence score. |
| **Plain English Writer** | Produces the two-part output: a short conversational answer (≤100 words) and a detailed structured breakdown. |
| **Evaluator** | Scores the Writer's output on accuracy, completeness, and citation quality. Triggers a refinement loop if score is below threshold. |

---

## 3. Data Flow

```
User query
  → Supervisor (check session memory, assess query clarity)
    → [if ambiguous] Clarifier → clarifying question → user response → Supervisor
    → Legal Researcher (RAG over irishstatutebook.ie)
    → Legal Analyst
    → Plain English Writer
    → Evaluator
      → [score < threshold] back to Legal Analyst / Writer (up to 2 refinement rounds)
      → [score ≥ threshold] return output to user
```

---

## 4. Session Memory

- Managed by the Supervisor using LangChain's `ConversationBufferMemory` (or equivalent)
- Scope: single session only (stateless across separate conversations)
- Tracks: conversation history, clarifying Q&A exchanges, topics explored in this session
- Passed as context to each agent on each turn so responses feel coherent

---

## 5. Structured Outputs

All agents return schema-validated JSON via LangChain's structured output / Pydantic models.

**Legal Researcher:**
```json
{
  "acts": [
    { "title": "string", "url": "string", "sections": ["string"] }
  ]
}
```
Semantic validation: `acts` must be non-empty.

**Legal Analyst:**
```json
{
  "key_clauses": ["string"],
  "gaps": ["string"],
  "confidence": 0.0
}
```
Semantic validation: `confidence` must be between 0.0 and 1.0.

**Plain English Writer:**
```json
{
  "short_answer": "string",
  "detailed_breakdown": {
    "summary": "string",
    "relevant_acts": ["string"],
    "key_clauses": ["string"],
    "caveats": ["string"]
  }
}
```
Semantic validation: `short_answer` must be ≤100 words.

**Evaluator:**
```json
{
  "score": 0.0,
  "flags": ["string"],
  "pass": true
}
```
Semantic validation: `score` between 0.0 and 1.0; `pass` is true if `score ≥ 0.7`.

---

## 6. Knowledge Base & RAG

- **Approach:** Live fetching from irishstatutebook.ie at query time
- **In-session cache:** Fetched pages cached in memory for the session duration to avoid duplicate requests
- **Retrieval:** The Legal Researcher searches the statute book site, fetches relevant Act pages, and extracts the pertinent sections
- **Future improvement:** Pre-indexed vector store for faster retrieval (out of scope for this project)

---

## 7. Error Handling

| Error Type | Handling |
|------------|----------|
| Transient (network, API timeout) | Retry with exponential backoff, max 3 attempts |
| Validation error (bad JSON, schema failure) | Regenerate up to 2 times, then fail safely |
| Fatal error | Return a graceful plain English message to the user explaining what went wrong |

---

## 8. Safety Guardrails

- Rate limiting on requests to irishstatutebook.ie
- Token budget cap per query (configurable)
- Max refinement rounds: 2 (prevents infinite loops)
- Evaluator pass threshold: 0.7 (configurable)

---

## 9. Multi-Agent Reasoning

- **Critique step:** Evaluator reviews Writer output and flags issues
- **Refinement loop:** Failed evaluation routes back to Analyst/Writer with the Evaluator's flags as context
- **Escalation:** If confidence from Analyst is below 0.4, Supervisor surfaces uncertainty to user in the final answer
- **Termination criteria:** Evaluator passes (score ≥ 0.7) OR max 2 refinement rounds reached

---

## 10. Deliverables Mapping

| Requirement | Implementation |
|-------------|----------------|
| 3+ agents | 6 agents (Supervisor, Clarifier, Researcher, Analyst, Writer, Evaluator) |
| Structured outputs | Pydantic-validated JSON per agent |
| Multi-agent reasoning | Critique + refinement loop, escalation on low confidence |
| Error handling | Retry, regenerate, fail safely |
| Safety guardrails | Rate limiting, token budget, loop caps |
| Knowledge base + memory | Live RAG from irishstatutebook.ie + session ConversationBufferMemory |
| Evaluation step | Evaluator agent with refinement loop |

---

## 11. Difficulty Level

**Gold** — semantic validation, memory, typed errors, evaluation loop, full guardrails, multi-round refinement, supervisor agent, cost tracking.
