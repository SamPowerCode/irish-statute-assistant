import sys
from irish_statute_assistant.config import Config
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
        *[f"  - {clause}" for clause in bd.key_clauses],
        "",
        "Things to be aware of:",
        *[f"  - {caveat}" for caveat in bd.caveats],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    config = Config()
    pipeline = Pipeline(config)

    print("Irish Statute Research Assistant")
    print("Type your legal question, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        try:
            result = pipeline.query(user_input)
            print(format_output(result))
        except Exception as e:
            print(f"\nSomething went wrong: {e}\nPlease try again or rephrase your question.\n")


if __name__ == "__main__":
    main()
