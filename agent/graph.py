"""Outer graph: decompose a possibly multi-hop question into self-contained
single-hop sub-questions, solve each with the Self-RAG subgraph
(agent/subgraph.py), then synthesize a final answer.

Usage (CLI):
    python -m agent.graph "阿尔托莉雅和贞德的宝具阶级哪个更高？"
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, StateGraph

from agent.llm import get_llm
from agent.schemas import DecomposeQuery
from agent.state import GraphState
from agent.subgraph import build_subgraph

logger = logging.getLogger(__name__)

_subgraph = None


def _get_subgraph():
    global _subgraph
    if _subgraph is None:
        _subgraph = build_subgraph()
    return _subgraph


def decompose(state: GraphState) -> dict:
    llm = get_llm().with_structured_output(DecomposeQuery)
    result: DecomposeQuery = llm.invoke(
        [
            (
                "system",
                "判断这个关于FGO（Fate/Grand Order）从者的问题是否需要拆解为多个"
                "独立的单跳子问题才能完整回答（例如涉及多个从者，或需要分别查询"
                "再比较/组合的情况）。如果不需要拆解，返回只包含原问题的列表。",
            ),
            ("human", state["question"]),
        ]
    )
    # Trust sub_questions directly rather than gating on is_complex: structured
    # output fills fields in declaration order, so the model commits to
    # is_complex before it has "worked out" the decomposition in sub_questions,
    # making the two fields inconsistent in practice (observed: is_complex=False
    # alongside a correct multi-item sub_questions list).
    sub_questions = result.sub_questions if len(result.sub_questions) > 1 else [state["question"]]
    logger.info("decompose: question=%r -> %d sub-question(s): %s", state["question"], len(sub_questions), sub_questions)
    return {"sub_questions": sub_questions}


def solve_subquestions(state: GraphState) -> dict:
    subgraph = _get_subgraph()
    sub_answers = []
    sub_documents = []
    for i, sub_q in enumerate(state["sub_questions"], 1):
        t0 = time.perf_counter()
        initial_state = {
            "question": sub_q,
            "route": "",
            "servant_name": "",
            "class_hint": "",
            "documents": [],
            "generation": "",
            "retrieve_retries": 0,
            "generate_retries": 0,
        }
        result = subgraph.invoke(initial_state)
        elapsed = time.perf_counter() - t0
        logger.info(
            "solve_subquestions: [%d/%d] %r done in %.1fs (%d doc(s))",
            i,
            len(state["sub_questions"]),
            sub_q,
            elapsed,
            len(result["documents"]),
        )
        sub_answers.append(result["generation"])
        sub_documents.append(result["documents"])
    return {"sub_answers": sub_answers, "sub_documents": sub_documents}


def synthesize(state: GraphState) -> dict:
    if len(state["sub_questions"]) == 1:
        return {"final_answer": state["sub_answers"][0]}

    llm = get_llm(temperature=0.3)
    qa_pairs = "\n\n".join(
        f"子问题：{q}\n子回答：{a}" for q, a in zip(state["sub_questions"], state["sub_answers"])
    )
    prompt = (
        "请基于下面每个子问题的回答，综合给出对原始问题的完整回答。"
        "如果子回答之间存在比较关系，请明确给出比较结论。\n\n"
        f"原始问题：{state['question']}\n\n{qa_pairs}"
    )
    result = llm.invoke([("human", prompt)])
    logger.info("synthesize: combined %d sub-answer(s) into final answer (len=%d)", len(state["sub_questions"]), len(result.content))
    return {"final_answer": result.content}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("decompose", decompose)
    graph.add_node("solve_subquestions", solve_subquestions)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "solve_subquestions")
    graph.add_edge("solve_subquestions", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def answer(question: str) -> dict:
    t0 = time.perf_counter()
    logger.info("answer: question=%r", question)
    graph = build_graph()
    result = graph.invoke({"question": question})
    logger.info("answer: total elapsed %.1fs", time.perf_counter() - t0)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("agent").setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    result = answer(args.question)
    print(f"\n子问题: {result['sub_questions']}")
    print(f"\n最终回答:\n{result['final_answer']}")
