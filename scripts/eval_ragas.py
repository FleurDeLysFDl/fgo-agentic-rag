"""RAGAS evaluation of the Phase 2 Self-RAG single-hop subgraph.

Runs every question in eval/questions.json through agent/subgraph.py's
run_single_hop, collects (question, answer, retrieved contexts, reference
answer), and scores the run with four RAGAS metrics:
  - faithfulness: is every claim in the answer grounded in the retrieved
    contexts? (no reference needed)
  - answer_relevancy: does the answer actually address the question?
    (no reference needed)
  - context_precision: are the retrieved contexts ranked with the relevant
    ones first? (needs reference)
  - context_recall: do the retrieved contexts cover everything needed to
    produce the reference answer? (needs reference)

Usage:
    python scripts/eval_ragas.py
    python scripts/eval_ragas.py --questions eval/questions_bulk_content.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nest_asyncio

# nest_asyncio's monkeypatch breaks asyncio.timeout()'s task-tracking on this
# Python version (observed: "RuntimeError: Timeout should be used inside a
# task"), and it isn't needed here since we run as a plain top-level script
# with no pre-existing event loop (unlike Jupyter, which is what it's for).
nest_asyncio.apply = lambda *a, **k: None

from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from agent.llm import get_llm
from agent.subgraph import run_single_hop
from config import LLM_API_BASE, LLM_API_KEY

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT_DIR / ".tmp" / "ragas_results.json"


def build_samples(questions_path: Path) -> list[dict]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    samples = []
    for q in questions:
        result = run_single_hop(q["question"])
        samples.append(
            {
                "user_input": q["question"],
                "response": result["generation"],
                "retrieved_contexts": [d["text"] for d in result["documents"]] or [""],
                "reference": q["ground_truth"],
            }
        )
        print(f"  done: {q['question']}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT_DIR / "eval" / "questions.json"))
    args = parser.parse_args()
    questions_path = Path(args.questions)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    print(f"Running {len(questions)} questions through the subgraph...")
    samples = build_samples(questions_path)

    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    )

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    df.to_json(RESULTS_PATH, orient="records", force_ascii=False, indent=2)

    print("\n=== Per-question scores ===")
    for _, row in df.iterrows():
        print(
            f"{row['user_input'][:30]:30s} "
            f"faith={row['faithfulness']:.2f} "
            f"ans_rel={row['answer_relevancy']:.2f} "
            f"ctx_prec={row['context_precision']:.2f} "
            f"ctx_rec={row['context_recall']:.2f}"
        )

    print("\n=== Averages ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"{metric}: {df[metric].mean():.4f}")


if __name__ == "__main__":
    main()
