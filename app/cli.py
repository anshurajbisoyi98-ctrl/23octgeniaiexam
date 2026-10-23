"""Ask a question from the terminal:  python -m app.cli "your question" """

import argparse

from app.rag import answer_question, format_result, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the GER-3620 maintenance assistant.")
    parser.add_argument("question", help="Your question in quotes")
    parser.add_argument("--debug", action="store_true", help="Show retrieved chunks and scores")
    args = parser.parse_args()

    if args.debug:
        for doc, score in retrieve(args.question, 5):
            print(f"{score:.3f}  p.{doc.metadata['page']:<3} {doc.metadata['section']}")
        print()
    print(format_result(answer_question(args.question)))


if __name__ == "__main__":
    main()
