from __future__ import annotations

import logging
import sys
from irish_statute_assistant.config import Config
from irish_statute_assistant.exceptions import (
    BudgetExceededError,
    StatuteNotFoundError,
    ValidationRepairError,
)
from irish_statute_assistant.models.schemas import WriterOutput
from irish_statute_assistant.pipeline import Pipeline


def format_output(result: WriterOutput | str) -> str:
    if isinstance(result, str):
        return f"\nI need a bit more information:\n  {result}\n"

    bd = result.detailed_breakdown
    lines = [
        "",
        f"Answer: {result.short_answer}",
        "",
        "--- Detail ---",
        f"Summary: {bd.summary}",
        "",
        "Relevant Acts:",
        *[f"  - {act}" for act in bd.relevant_acts],
        "",
        "Key points:",
        *[f"  - {kc.text} ({kc.act}, {kc.section})" for kc in bd.key_clauses],
        "",
        "Things to be aware of:",
        *[f"  - {caveat}" for caveat in bd.caveats],
        "",
    ]

    if result.analyst_confidence < 0.5:
        lines.append("Note: confidence in statute coverage was low for this query.")
        lines.append("")

    if result.warnings:
        lines.append("--- Grounding warnings ---")
        lines.extend(f"  - {w}" for w in result.warnings)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    config = Config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s — %(message)s",
    )
    pipeline = Pipeline(config)

    print("Irish Statute Research Assistant")
    print(
        "DISCLAIMER: This tool is not a substitute for legal advice. It provides an\n"
        "interpretation of statute text only and does not account for upcoming or\n"
        "recently enacted legislation, or how the courts have interpreted these statutes.\n"
        "For legal matters, consult a qualified solicitor or barrister.\n"
    )
    print("Type your legal question, or 'quit' to exit.\n")

    pending_query: str | None = None
    while True:
        prompt = "Your clarification: " if pending_query else "Your question: "
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        if pending_query:
            pipeline_input = f"{pending_query}\n\n[User clarification: {user_input}]"
            pending_query = None
        else:
            pipeline_input = user_input

        try:
            result = pipeline.query(pipeline_input)
            if isinstance(result, str):
                pending_query = pipeline_input
            print(format_output(result))
        except StatuteNotFoundError:
            print("\nNo relevant statutes were found for your question. Please try a different topic.\n")
        except BudgetExceededError:
            print("\nThis query exceeded the token budget. You can increase TOKEN_BUDGET_PER_QUERY in your .env file.\n")
        except ValidationRepairError:
            print("\nThe assistant could not produce a valid response after several attempts. Please try rephrasing.\n")
        except Exception as e:
            print(f"\nSomething went wrong: {e}\nPlease try again or rephrase your question.\n")


if __name__ == "__main__":
    main()
