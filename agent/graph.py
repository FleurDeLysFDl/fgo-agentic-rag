"""Outer graph: resolve the question against conversation history (or ask for
clarification if it can't be resolved) -> decompose a possibly multi-hop
question into self-contained single-hop sub-questions -> solve each with the
Self-RAG subgraph (agent/subgraph.py) -> synthesize a final answer.

Either stage can short-circuit straight to a clarifying question instead of
an answer: resolve_question if a pronoun/reference can't be resolved even
with history (or the question is missing essential scope), or
solve_subquestions if a sub-question's retrieved documents disagree with
each other on the fact being asked about (agent/subgraph.py's
check_conflict). Callers don't need to branch on this -- result["final_answer"]
is always the right thing to show the user; result["needs_clarification"]
just tells the UI whether it's a question rather than an answer (e.g. to
skip a "sources" section).

Callers should pass clarification_rounds = the number of consecutive
assistant turns that were clarification-only (no real answer) so far in
this conversation. Once that hits MAX_CLARIFICATION_ROUNDS (agent/state.py),
resolve_question stops asking and commits to a best-effort interpretation,
guaranteeing the conversation converges on an answer instead of narrowing
forever.

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
from agent.schemas import DecomposeQuery, ResolvedQuestion
from agent.state import MAX_CLARIFICATION_ROUNDS, GraphState, Turn
from agent.subgraph import build_subgraph

logger = logging.getLogger(__name__)

_subgraph = None


def _get_subgraph():
    global _subgraph
    if _subgraph is None:
        _subgraph = build_subgraph()
    return _subgraph


def resolve_question(state: GraphState) -> dict:
    history = state.get("history") or []
    history_text = "\n".join(
        f"{'用户' if turn['role'] == 'user' else '助手'}：{turn['content']}" for turn in history
    )
    clarification_rounds = state.get("clarification_rounds", 0)

    if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        # Already asked this many rounds without landing on an answer -- an
        # open-ended question can always be narrowed further, so without a
        # hard stop this can loop indefinitely (observed: 8 rounds narrowing
        # a relationship question with no answer ever generated). Force a
        # best-effort self-contained question instead of asking again.
        result = get_llm().invoke(
            [
                (
                    "system",
                    "已经反问用户好几轮但还没有给出实际回答。请结合完整对话历史和用户"
                    "刚才这句话，直接给出一个自包含、消解了所有代词/指代的问题，不要再"
                    "反问——用你能想到的最合理解读。只输出改写后的问题本身，不要输出多余内容。",
                ),
                ("human", f"对话历史：\n{history_text or '（无历史）'}\n\n当前这句话：{state['question']}"),
            ]
        )
        resolved = result.content.strip() or state["question"]
        logger.info(
            "resolve_question: clarification_rounds=%d >= max, forcing resolution -> %r",
            clarification_rounds,
            resolved,
        )
        return {"needs_clarification": False, "question": resolved}

    llm = get_llm().with_structured_output(ResolvedQuestion)
    result: ResolvedQuestion = llm.invoke(
        [
            (
                "system",
                "你是FGO（Fate/Grand Order）问答系统的问题理解模块。结合对话历史，判断"
                "当前问题是否包含无法确定所指对象的代词/指代（如'她'、'这个从者'，且历史中"
                "找不到可以对应的对象）或缺失回答所必需的关键信息。如果历史信息足够，把问题"
                "改写为不依赖上下文、独立完整的问题（消解代词/指代），保持原语言；如果问题"
                "本身已经independent，原样返回。如果历史信息不够，提出一个简短的反问来获取"
                "缺失信息，语言与原问题一致。",
            ),
            ("human", f"对话历史：\n{history_text or '（无历史）'}\n\n当前问题：{state['question']}"),
        ]
    )
    logger.info(
        "resolve_question: question=%r -> needs_clarification=%s resolved_question=%r",
        state["question"],
        result.needs_clarification,
        result.resolved_question,
    )
    if result.needs_clarification:
        return {
            "needs_clarification": True,
            "clarification_question": result.clarification_question,
            "final_answer": result.clarification_question,
        }
    return {
        "needs_clarification": False,
        "question": result.resolved_question or state["question"],
    }


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
    clarifications = []
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
            "needs_clarification": False,
            "clarification_question": "",
            "clarification_rounds": state.get("clarification_rounds", 0),
        }
        result = subgraph.invoke(initial_state)
        elapsed = time.perf_counter() - t0

        if result.get("needs_clarification"):
            logger.info(
                "solve_subquestions: [%d/%d] %r needs clarification (%.1fs)",
                i,
                len(state["sub_questions"]),
                sub_q,
                elapsed,
            )
            clarifications.append(result["clarification_question"])
            sub_answers.append(result["clarification_question"])
            sub_documents.append([])
            continue

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

    if clarifications:
        combined = (
            clarifications[0]
            if len(clarifications) == 1
            else "在回答之前，我需要先确认几点：\n\n" + "\n\n".join(clarifications)
        )
        return {
            "sub_answers": sub_answers,
            "sub_documents": sub_documents,
            "needs_clarification": True,
            "clarification_question": combined,
            "final_answer": combined,
        }
    return {"sub_answers": sub_answers, "sub_documents": sub_documents, "needs_clarification": False}


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
    graph.add_node("resolve_question", resolve_question)
    graph.add_node("decompose", decompose)
    graph.add_node("solve_subquestions", solve_subquestions)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("resolve_question")

    def after_resolve_question(state: GraphState) -> str:
        return "clarify" if state.get("needs_clarification") else "decompose"

    graph.add_conditional_edges(
        "resolve_question", after_resolve_question, {"clarify": END, "decompose": "decompose"}
    )
    graph.add_edge("decompose", "solve_subquestions")

    def after_solve_subquestions(state: GraphState) -> str:
        return "clarify" if state.get("needs_clarification") else "synthesize"

    graph.add_conditional_edges(
        "solve_subquestions", after_solve_subquestions, {"clarify": END, "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


def answer(question: str, history: list[Turn] | None = None, clarification_rounds: int = 0) -> dict:
    t0 = time.perf_counter()
    logger.info(
        "answer: question=%r history_len=%d clarification_rounds=%d",
        question,
        len(history or []),
        clarification_rounds,
    )
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "history": history or [],
        "clarification_rounds": clarification_rounds,
        "needs_clarification": False,
        "clarification_question": "",
        "sub_questions": [],
        "sub_answers": [],
        "sub_documents": [],
        "final_answer": "",
    }
    result = graph.invoke(initial_state)
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
    if result["needs_clarification"]:
        print(f"\n需要澄清:\n{result['final_answer']}")
    else:
        print(f"\n子问题: {result['sub_questions']}")
        print(f"\n最终回答:\n{result['final_answer']}")
