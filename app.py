"""Streamlit demo UI for the Irish Statute Research Assistant.

Layout:
  Sidebar — live pipeline trace (sticky, always visible while scrolling)
  Main    — chat conversation (questions + answers)

Run with:
    uv run streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

from irish_statute_assistant.config import Config
from irish_statute_assistant.exceptions import (
    BudgetExceededError,
    FatalError,
    StatuteNotFoundError,
    ValidationRepairError,
)
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Irish Statute Research Assistant",
    page_icon="🏛",
    layout="wide",
)

# ── Pipeline (one instance, shared across reruns) ─────────────────────────────

@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(Config())

pipeline = get_pipeline()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []        # {"role": "user"|"assistant", "content": str}
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []  # {"agent": str, "stats": dict}
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None  # original query awaiting clarification

# Clear the pipeline steps as soon as a new query is submitted — before the
# sidebar renders — so the old steps don't flash while the new query runs.
# st.chat_input stores its value in session_state[key] at the start of the
# rerun, before the widget call itself, so this check works.
if st.session_state.get("_chat_input"):
    st.session_state.pipeline_steps = []

# ── Helper: render one pipeline step row ──────────────────────────────────────

def _step_label(agent: str, stats: dict) -> str:
    """Return a markdown string for one completed pipeline step row."""
    if agent == "Devil's Advocate":
        icon = "⚔️"
        round_num = stats.get("round", 0)
        label = f"Devil's Advocate" + (f" (round {round_num})" if round_num else " (initial)")
    elif agent == "Clarifier" and stats.get("needs_clarification"):
        icon = "❓"
        label = "Clarifier"
    else:
        icon = "✅"
        label = agent

    dur = stats.get("duration_s", 0)

    if agent == "Clarifier":
        detail = "needs clarification" if stats.get("needs_clarification") else "no clarification needed"
    elif agent == "Researcher":
        detail = f"{stats.get('acts_found', 0)} acts · {stats.get('source', '')}"
    elif agent == "Analyst":
        detail = f"{stats.get('key_clauses', 0)} clauses · confidence {stats.get('confidence', 0):.2f}"
    elif agent == "Devil's Advocate":
        detail = f"{stats.get('challenges', 0)} challenge(s) · {stats.get('severity', '')}"
    elif agent == "Writer":
        detail = f"round {stats.get('round', 1)}"
    elif agent == "Grounding Checker":
        detail = "all claims grounded" if stats.get("grounding_passed") else f"{stats.get('ungrounded', 0)} ungrounded"
    elif agent == "Evaluator":
        outcome = "passed ✓" if stats.get("passed") else "failed ✗"
        detail = f"score {stats.get('score', 0):.2f} · {outcome}"
    else:
        detail = ""

    return f"{icon} **{label}** — {detail} *({dur}s)*"


def _render_pipeline(steps: list[dict], spinning: str | None = None) -> None:
    """Render pipeline steps into the current Streamlit column context."""
    if not steps and spinning is None:
        st.caption("Pipeline trace will appear here during a query.")
        return

    for step in steps:
        st.markdown(_step_label(step["agent"], step["stats"]))

    if spinning:
        st.markdown(f"⏳ **{spinning}**…")

    if steps and spinning is None:
        total_dur = sum(s["stats"].get("duration_s", 0) for s in steps)
        rounds = max(
            (s["stats"].get("round", 1) for s in steps if s["agent"] == "Writer"),
            default=1,
        )
        st.divider()
        st.caption(f"{rounds} round(s) · {total_dur:.1f}s total")


# ── Sidebar: pipeline trace ────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Pipeline")
    pipe_placeholder = st.empty()
    with pipe_placeholder.container():
        _render_pipeline(st.session_state.pipeline_steps)

# ── Render existing conversation ───────────────────────────────────────────────

st.title("🏛 Irish Statute Research Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a legal question…", key="_chat_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # If we're in a clarification flow, combine the original query with the user's answer
    if st.session_state.pending_query:
        pipeline_input = f"{st.session_state.pending_query}\n\n[User clarification: {user_input}]"
        st.session_state.pending_query = None
    else:
        pipeline_input = user_input

    # Track next expected agent for the spinner
    _AGENT_ORDER = [
        "Clarifier", "Researcher", "Analyst", "Devil's Advocate",
        "Writer", "Grounding Checker", "Evaluator",
    ]
    next_agent: list[str] = ["Clarifier"]

    def on_step(agent_name: str, stats: dict) -> None:
        st.session_state.pipeline_steps.append({"agent": agent_name, "stats": stats})
        # Advance spinner to the next expected agent (rough heuristic)
        try:
            idx = _AGENT_ORDER.index(agent_name)
            next_agent[0] = _AGENT_ORDER[idx + 1] if idx + 1 < len(_AGENT_ORDER) else ""
        except ValueError:
            next_agent[0] = ""
        with pipe_placeholder.container():
            _render_pipeline(st.session_state.pipeline_steps, spinning=next_agent[0] or None)

    # Show initial spinner
    with pipe_placeholder.container():
        _render_pipeline([], spinning="Clarifier")

    result = None
    try:
        result = pipeline.query(pipeline_input, progress_callback=on_step)
    except StatuteNotFoundError:
        st.error("No relevant statutes found. Please try rephrasing your question.")
    except BudgetExceededError:
        st.error(
            "This query exceeded the token budget. "
            "Increase `TOKEN_BUDGET_PER_QUERY` in your `.env` file."
        )
    except ValidationRepairError:
        st.error("Could not produce a valid response after several attempts. Please try rephrasing.")
    except FatalError as e:
        st.error(f"An unrecoverable error occurred. Check the terminal for details. ({e})")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

    # Final render — no spinner
    with pipe_placeholder.container():
        _render_pipeline(st.session_state.pipeline_steps)

    if isinstance(result, str):
        # Clarifying question — remember the original query so the next response has context
        st.session_state.pending_query = pipeline_input
        st.session_state.messages.append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.markdown(result)
    elif isinstance(result, WriterOutput):
        bd = result.detailed_breakdown
        lines = [f"**{result.short_answer}**", ""]

        if bd.relevant_acts:
            lines.append("**Relevant Acts:** " + ", ".join(bd.relevant_acts))
            lines.append("")

        if bd.key_clauses:
            lines.append("**Key clauses:**")
            for kc in bd.key_clauses:
                lines.append(f"- {kc.text} *({kc.act}, {kc.section})*")
            lines.append("")

        if bd.caveats:
            lines.append("**Things to be aware of:**")
            for caveat in bd.caveats:
                lines.append(f"- {caveat}")

        if result.warnings:
            lines.append("")
            lines.append("⚠️ *Some claims could not be verified against source text.*")

        if result.analyst_confidence < 0.5:
            lines.append("")
            lines.append("*Note: confidence in statute coverage was low for this query.*")

        content = "\n".join(lines)
        st.session_state.messages.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"):
            st.markdown(content)
