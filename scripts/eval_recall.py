"""Recall@5 baseline evaluation for the hybrid retrieval pipeline.

Runs each question in a questions file (default eval/questions.json, the
25 hand-written/verified questions) through HybridRetriever.
query_verbose(top_k=5) and checks whether the expected source document
appears among the top-5 results. Every pipeline stage (dense search, BM25
search, RRF fusion, cross-encoder rerank) is printed so the run can be
watched live.

Usage:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --questions eval/questions_bulk.json --log eval/recall_results_bulk.log
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import HybridRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent

SEP = "-" * 70


def make_tee(log_file):
    """Console output can be piped through PowerShell (which mangles UTF-8
    CJK text via its own re-encoding), so write the UTF-8 log file directly
    from Python instead of relying on shell redirection."""

    def tee(text: str = "") -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
        log_file.write(text + "\n")
        log_file.flush()

    return tee


def print_stage(tee, title: str, lines: list[str]) -> None:
    tee(f"  [{title}]")
    for line in lines:
        tee(f"    {line}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT_DIR / "eval" / "questions.json"))
    parser.add_argument("--log", default=str(ROOT_DIR / "eval" / "recall_results.log"))
    args = parser.parse_args()
    questions_path = Path(args.questions)
    log_path = Path(args.log)

    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = make_tee(log_file)

        tee("Loading retriever (embedding + reranker models, BM25 index, Qdrant)...")
        retriever = HybridRetriever()
        tee("Retriever ready.\n")

        hits = 0
        for i, q in enumerate(questions, start=1):
            tee(SEP)
            tee(f"[{i}/{len(questions)}] Q: {q['question']}")
            tee(f"  期望来源: {q['expected_source']}")

            result = retriever.query_verbose(q["question"], top_k=5)

            print_stage(
                tee,
                "1. Dense search (bge-m3, top-20)",
                [f"{h['score']:.4f}  {h['source']}" for h in result["dense_hits"][:5]] + ["..."],
            )
            print_stage(
                tee,
                "2. BM25 search (jieba tokens, top-20)",
                [f"{h['score']:.4f}  {h['source']}" for h in result["bm25_hits"][:5]] + ["..."],
            )
            print_stage(
                tee,
                "3. RRF fusion (top-20 sent to reranker)",
                [f"{rank}. {src}" for rank, src in enumerate(result["fused"][:5], start=1)] + ["..."],
            )
            print_stage(
                tee,
                "4. Cross-encoder rerank (bge-reranker-v2-m3)",
                [f"{h['score']:.4f}  {h['source']}" for h in result["reranked"][:5]],
            )

            final_sources = [r["source"] for r in result["final"]]
            hit = q["expected_source"] in final_sources
            hits += hit
            status = "命中" if hit else "未命中"
            print_stage(
                tee,
                f"5. FINAL top-5 -> {status}",
                [f"{r['rerank_score']:.4f}  {r['source']}" for r in result["final"]],
            )

        recall_at_5 = hits / len(questions)
        tee(SEP)
        tee(f"\n=== Recall@5 = {hits}/{len(questions)} = {recall_at_5:.2%} ===")


if __name__ == "__main__":
    main()
