"""Recall@5 baseline evaluation for the hybrid retrieval pipeline.

Runs each hand-written question in eval/questions.json through
HybridRetriever.query(top_k=5) and checks whether the expected source
document appears among the top-5 results.

Usage:
    python scripts/eval_recall.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import HybridRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT_DIR / "eval" / "questions.json"


def main() -> None:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        questions = json.load(f)

    retriever = HybridRetriever()

    hits = 0
    for i, q in enumerate(questions, start=1):
        results = retriever.query(q["question"], top_k=5)
        sources = [r["source"] for r in results]
        hit = q["expected_source"] in sources
        hits += hit
        status = "命中" if hit else "未命中"
        print(f"[{i}/{len(questions)}] {status} | Q: {q['question']}")
        print(f"    期望来源: {q['expected_source']}")
        print(f"    Top-5来源: {sources}")

    recall_at_5 = hits / len(questions)
    print(f"\n=== Recall@5 = {hits}/{len(questions)} = {recall_at_5:.2%} ===")


if __name__ == "__main__":
    main()
