"""Run every question in eval/questions.json and write eval/results.md and eval/results.json.

Run with:  python -m eval.run_eval
"""

import json
from pathlib import Path

from app.config import get_settings
from app.rag import answer_question, format_result

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    questions = json.loads((EVAL_DIR / "questions.json").read_text())
    settings = get_settings()
    rows = []
    for item in questions:
        result = answer_question(item["question"])
        rows.append({
            **item,
            "found": result.found,
            "correct": result.found == item["expected_found"],
            "output": format_result(result),
            "citations": [c.model_dump() for c in result.citations],
            "top_score": round(result.top_score, 3),
            "latency_ms": result.latency_ms,
        })
        print(f"[{item['id']}] {item['question']}\n{rows[-1]['output']}\n")

    (EVAL_DIR / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    passed = sum(r["correct"] for r in rows)
    lines = [
        "# Evaluation results",
        "",
        f"LLM: `{settings.llm_provider}` / `{settings.gemini_model if settings.llm_provider == 'gemini' else settings.groq_model}` · "
        f"embeddings: `{settings.embedding_model}` · top_k={settings.top_k} · "
        f"threshold={settings.relevance_threshold}",
        "",
        f"**{passed}/{len(rows)}** questions gave the expected found / not-found result.",
        "",
        "| # | Question | Expected | Top score | System output |",
        "|---|----------|----------|-----------|---------------|",
    ]
    for r in rows:
        output = r["output"].replace("\n", "<br>").replace("|", "\\|")
        expected = "answer" if r["expected_found"] else "Not found"
        mark = "✅" if r["correct"] else "❌"
        lines.append(f"| {r['id']} | {r['question']} | {expected} {mark} | {r['top_score']} | {output} |")
    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print(f"{passed}/{len(rows)} correct. Wrote eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
