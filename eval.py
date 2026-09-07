"""
Stage 5 — Score
In:  chunks_<video_id>.json (from main.py) + questions.json (frozen, hand-written)
Out: results.json — every question, the system's answer, and whether it was
     graded correct, plus a per-category summary table printed to stdout.

Grading is manual by design: you read each answer against your own
expected_answer_notes and mark it. This script records your judgment; it
doesn't try to auto-grade text similarity, which would just be a second
unverified system grading the first one.
"""

import sys
import json
from pathlib import Path

from retrieve import Retriever
from answer import answer_question


def run_eval(chunks_path: Path, questions_path: Path, results_path: Path):
    chunks_data = json.loads(chunks_path.read_text())
    questions_data = json.loads(questions_path.read_text())

    retriever = Retriever(chunks_data["chunks"])
    results = []

    for q in questions_data["questions"]:
        print(f"\n[{q['id']}] ({q['kind']}) {q['question']}")
        result = answer_question(q["question"], retriever)
        print(f"  -> {result['answer']}")

        grade = input("  Correct? (y/n/skip): ").strip().lower()
        results.append({
            **q,
            "system_answer": result["answer"],
            "reason": result["reason"],
            "top_score": result["top_score"],
            "graded_correct": grade == "y" if grade in ("y", "n") else None,
        })

    results_path.write_text(json.dumps(results, indent=2))

    # summary by kind
    by_kind = {}
    for r in results:
        kind = r["kind"]
        by_kind.setdefault(kind, {"correct": 0, "total": 0, "skipped": 0})
        if r["graded_correct"] is None:
            by_kind[kind]["skipped"] += 1
        else:
            by_kind[kind]["total"] += 1
            if r["graded_correct"]:
                by_kind[kind]["correct"] += 1

    print("\n--- Summary ---")
    total_correct, total_scored = 0, 0
    for kind, stats in by_kind.items():
        print(f"{kind:12s}: {stats['correct']}/{stats['total']} correct"
              f" ({stats['skipped']} skipped)")
        total_correct += stats["correct"]
        total_scored += stats["total"]
    print(f"{'TOTAL':12s}: {total_correct}/{total_scored}")
    print(f"\nFull results -> {results_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval.py <chunks_<video_id>.json> <questions.json> [results.json]")
        sys.exit(1)

    chunks_path = Path(sys.argv[1])
    questions_path = Path(sys.argv[2])
    results_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results.json")

    run_eval(chunks_path, questions_path, results_path)


if __name__ == "__main__":
    main()
