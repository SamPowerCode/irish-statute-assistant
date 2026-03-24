# Agentic AI Final Project Brief

**Creative Multi-Agent Intelligent System — 7-Week Capstone**

---

## 1. Project Objective

You will design and build a production-grade multi-agent AI system that can collaborate, reason, and reliably complete a complex task within a world or theme of your choosing. The focus is on technical reliability, structured reasoning, and safe autonomous behaviour.

---

## 2. Core Requirements (Mandatory)

### A. Minimum of Three Agents

Your system must contain at least three distinct agents with clear roles. Agents must communicate, exchange information, and refine each other's outputs.

### B. Structured Outputs

All agents must return schema-validated JSON. Schemas must include semantic validation, not only type validation.

### C. Multi-Agent Reasoning

Agents must demonstrate critique steps, multi-step reasoning loops, escalation when uncertain, and clear termination criteria.

### D. Error Handling

Your system must handle:
- Transient errors (retry with backoff)
- Validation errors (repair or regenerate)
- Fatal errors (fail safely)

### E. Safety Guardrails

Your system must include rate limiting, budget control, etc.

### F. Knowledge & Memory

Use a knowledge base or retrieval component and maintain short- or long-term memory of user preferences.

### G. Evaluation Step

Include an evaluator that scores outputs for correctness, structure, completeness, and citations, with at least one refinement loop.

---

## 3. Creative Freedom

You may choose any world — sci-fi, fantasy, business, education, cyber-security, or one you invent. Creativity is welcome, but technical reliability remains the priority.

---

## 4. Deliverables (Week 7 – Demo Day)

1. Working system (live, new scenario)
2. Architecture diagram (screenshot)
3. Trace logs (success + recovery)
4. Evaluation pack
5. Final outputs (JSON + rendered)
6. 3-minute demo

---

## 5. Difficulty Levels (Self-Selection)

| Level | Requirements |
|-------|-------------|
| **Bronze** | 3 agents, basic schemas, simple KB, basic guardrails |
| **Silver** | Semantic validation, memory, typed errors, evaluation loop |
| **Gold** | Full guardrails, multi-round refinement, supervisor agent, cost tracking, adversarial tests |
| **Platinum** | Debate-style reasoning, auto-revision, hallucination detection, advanced constraints, multiple memory stores |
