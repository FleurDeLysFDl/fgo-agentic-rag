"""Single-hop Self-RAG-style subgraph: route -> retrieve/structured-lookup ->
(retry with rewritten query if nothing came back) -> generate -> grade
generation (hallucination + answer-quality) -> retry generate or retry
retrieval if needed, else return.

If structured_lookup's servant name/class don't pin down a single row (e.g.
"阿尔托莉雅的宝具是什么" with no class specified matches several playable
forms), the subgraph short-circuits with needs_clarification=True/
clarification_question instead of guessing or blending facts from multiple
variants into one answer -- see structured_lookup_node.

Retrieved/looked-up documents are NOT graded for relevance with a per-document
LLM call -- that was the dominant cost in wall-clock latency (one sequential
LLM round-trip per candidate, e.g. 10 calls for a 10-match structured lookup).
Vector-retrieved candidates instead get a free relevance gate:
_select_by_score_gap cuts the sorted candidate list at its steepest
cross-encoder-score drop (always keeping at least MIN_KEPT_DOCUMENTS) before
it ever reaches generate(), without an LLM call. structured_lookup has no
such score (it's a name match, not a ranked search) so its candidates pass
through unfiltered.
Hallucination/answer-quality grading after generation is unaffected -- those
are single calls regardless of candidate count.

gpt-4o-mini has not been fine-tuned with literal Self-RAG reflection tokens,
so each "reflection" judgment (retrieve-worthy? supported? useful?) is
elicited via structured LLM output (agent/schemas.py) instead of special
vocabulary tokens -- functionally equivalent for a general-purpose chat model.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import END, StateGraph

from agent.llm import get_llm
from agent.retriever_singleton import get_retriever
from agent.schemas import (
    GradeAnswer,
    GradeHallucination,
    RewrittenQuery,
    RouteQuery,
)
from agent.state import SubState
from agent.structured_lookup import lookup_servant

logger = logging.getLogger(__name__)

MAX_RETRIEVE_RETRIES = 2
MAX_GENERATE_RETRIES = 2
# With per-document LLM grading removed, this is the only remaining relevance
# gate on vector-retrieved candidates. A fixed score cutoff doesn't fit the
# data: the gap between "actually relevant" and "noise" candidates is real
# (e.g. observed rank-1/2 hits at 0.98/0.66 next to rank-3+ noise at
# 0.02/0.004/0.003) but its absolute position moves per query, so instead of
# a fixed threshold this cuts at the single steepest score drop in the sorted
# list (see _select_by_score_gap) -- wherever the biggest cliff actually is
# for that query -- while always keeping at least MIN_KEPT_DOCUMENTS so a
# query with no sharp cliff (every candidate plausibly relevant) doesn't get
# left with just one.
MIN_KEPT_DOCUMENTS = 2


def _select_by_score_gap(results: list[dict], min_keep: int = MIN_KEPT_DOCUMENTS) -> list[dict]:
    """Keep the top-scoring results up through the steepest drop in
    consecutive rerank_score values (results are already sorted descending),
    always keeping at least min_keep regardless of where that drop falls."""
    if len(results) <= min_keep:
        return results
    scores = [r["rerank_score"] for r in results]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    cliff = max(range(len(gaps)), key=lambda i: gaps[i]) + 1  # keep up through the biggest drop
    return results[: max(cliff, min_keep)]


def _format_structured_doc(servant: dict) -> dict:
    skills_text = "\n".join(f"  - {s['name_en']}：{s['detail_en']}" for s in servant["skills"])
    nps_text = "\n".join(
        f"  - {np['name_en']}（卡色：{np['card_type']}，阶级：{np['rank']}）：{np['detail_en']}"
        for np in servant["noble_phantasms"]
    )
    text = (
        f"从者：{servant['name_cn']}（{servant['name_en']}）\n"
        f"职阶：{servant['class_name']}　稀有度：{servant['rarity']}星\n"
        f"获取途径：{servant['acquisition'] or '未知'}\n"
        f"技能：\n{skills_text or '  （无数据）'}\n"
        f"宝具：\n{nps_text or '  （无数据）'}"
    )
    return {"text": text, "source": f"{servant['name_cn']}（{servant['class_name']}）"}


def route_question(state: SubState) -> dict:
    llm = get_llm().with_structured_output(RouteQuery)
    result: RouteQuery = llm.invoke(
        [
            (
                "system",
                "你是FGO（Fate/Grand Order）问答系统的路由模块。判断问题应该查询"
                "结构化数据库（技能、宝具卡色/阶级、稀有度、职阶、获取途径等游戏数值类事实）"
                "还是向量语料库（背景故事、性格、羁绊等叙事类内容）。",
            ),
            ("human", state["question"]),
        ]
    )
    logger.info(
        "route_question: question=%r -> route=%s servant_name=%r class_hint=%r",
        state["question"],
        result.route,
        result.servant_name,
        result.class_hint,
    )
    return {"route": result.route, "servant_name": result.servant_name, "class_hint": result.class_hint}


def structured_lookup_node(state: SubState) -> dict:
    matches = lookup_servant(state["servant_name"], class_hint=state.get("class_hint") or None)

    if len(matches) > 1:
        # name_cn isn't unique across class-swap/costume variants (see
        # lookup_servant's docstring) -- more than one match means the
        # question's servant name/class don't pin down a single servant, so
        # ask rather than silently answering for whichever row happened to
        # come back first (or worse, mixing facts from multiple variants
        # into one answer, as observed before this check existed).
        variant_names = sorted({f"{m['name_cn']}（{m['class_name']}）" for m in matches})
        clarification = (
            f"「{state['servant_name']}」匹配到{len(variant_names)}个不同的从者形态，"
            "请问具体是指哪一个？\n" + "\n".join(f"- {v}" for v in variant_names)
        )
        logger.info(
            "structured_lookup: servant_name=%r class_hint=%r -> %d match(es), ambiguous -- asking for clarification",
            state["servant_name"],
            state.get("class_hint"),
            len(matches),
        )
        return {"documents": [], "needs_clarification": True, "clarification_question": clarification}

    documents = [_format_structured_doc(m) for m in matches]
    logger.info(
        "structured_lookup: servant_name=%r class_hint=%r -> %d match(es)",
        state["servant_name"],
        state.get("class_hint"),
        len(matches),
    )
    return {"documents": documents, "needs_clarification": False}


def retrieve(state: SubState) -> dict:
    retriever = get_retriever()
    results = retriever.query(state["question"], top_k=5)
    kept = _select_by_score_gap(results)
    documents = [{"text": r["text"], "source": r["source"]} for r in kept]
    logger.info(
        "retrieve: query=%r -> %d/%d result(s) kept up to score cliff: %s",
        state["question"],
        len(documents),
        len(results),
        [(d["source"], round(r["rerank_score"], 4)) for d, r in zip(documents, kept)],
    )
    return {"documents": documents}


def decide_to_generate(state: SubState) -> str:
    if state["documents"]:
        return "generate"
    if state["retrieve_retries"] < MAX_RETRIEVE_RETRIES:
        logger.info(
            "decide_to_generate: no relevant docs, retrying (attempt %d/%d)",
            state["retrieve_retries"] + 1,
            MAX_RETRIEVE_RETRIES,
        )
        return "transform_query"
    logger.info("decide_to_generate: no relevant docs and retries exhausted, generating anyway")
    return "generate"


def decide_after_structured_lookup(state: SubState) -> str:
    if state.get("needs_clarification"):
        return "clarify"
    return decide_to_generate(state)


def transform_query(state: SubState) -> dict:
    llm = get_llm().with_structured_output(RewrittenQuery)
    result: RewrittenQuery = llm.invoke(
        [
            (
                "system",
                "之前的检索没有找到相关资料。请改写问题使其更利于检索"
                "（更清晰的实体名称、消除歧义、修正可能的错字），保持原意与原语言。",
            ),
            ("human", state["question"]),
        ]
    )
    logger.info("transform_query: %r -> %r", state["question"], result.better_question)
    return {
        "question": result.better_question,
        "retrieve_retries": state["retrieve_retries"] + 1,
    }


def generate(state: SubState) -> dict:
    llm = get_llm(temperature=0.3)
    if state["documents"]:
        context = "\n\n---\n\n".join(f"[来源：{d['source']}]\n{d['text']}" for d in state["documents"])
        prompt = (
            "请仅根据下面提供的资料回答问题，不要编造资料中没有的内容。"
            "回答完给出引用的来源名称。\n\n"
            f"资料：\n{context}\n\n问题：{state['question']}"
        )
    else:
        prompt = (
            "没有检索到与问题相关的资料。请直接告知用户无法在现有语料库中找到"
            f"关于这个问题的可靠信息，不要编造答案。\n\n问题：{state['question']}"
        )
    result = llm.invoke([("human", prompt)])
    logger.info(
        "generate: %d document(s) as context -> answer_len=%d",
        len(state["documents"]),
        len(result.content),
    )
    return {"generation": result.content}


def grade_generation(state: SubState) -> str:
    if not state["documents"]:
        logger.info("grade_generation: no documents to check against, accepting honest 'no info' answer")
        return "useful"  # nothing to hallucinate against; already an honest "no info" answer

    llm_hallucination = get_llm().with_structured_output(GradeHallucination)
    context = "\n\n---\n\n".join(d["text"] for d in state["documents"])
    hallucination: GradeHallucination = llm_hallucination.invoke(
        [
            ("system", "判断生成的回答中的每一个论述是否都有资料支持，只回答yes或no。"),
            ("human", f"资料：{context}\n\n生成的回答：{state['generation']}"),
        ]
    )
    if hallucination.binary_score == "no":
        if state["generate_retries"] < MAX_GENERATE_RETRIES:
            logger.info(
                "grade_generation: hallucination detected, retrying generation (attempt %d/%d)",
                state["generate_retries"] + 1,
                MAX_GENERATE_RETRIES,
            )
            return "not supported"
        logger.info("grade_generation: hallucination detected but retries exhausted, returning best-effort answer")
        return "useful"  # stop retrying, return best-effort answer rather than loop forever

    llm_answer = get_llm().with_structured_output(GradeAnswer)
    answer: GradeAnswer = llm_answer.invoke(
        [
            ("system", "判断生成的回答是否切实回答了用户的问题，只回答yes或no。"),
            ("human", f"问题：{state['question']}\n\n回答：{state['generation']}"),
        ]
    )
    if answer.binary_score == "no" and state["retrieve_retries"] < MAX_RETRIEVE_RETRIES:
        logger.info("grade_generation: answer judged not useful, retrying retrieval")
        return "not useful"
    logger.info("grade_generation: answer judged useful, done")
    return "useful"


def increment_generate_retries(state: SubState) -> dict:
    return {"generate_retries": state["generate_retries"] + 1}


def build_subgraph():
    graph = StateGraph(SubState)

    graph.add_node("route_question", route_question)
    graph.add_node("structured_lookup", structured_lookup_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("increment_generate_retries", increment_generate_retries)

    graph.set_entry_point("route_question")

    def route_branch(state: SubState) -> str:
        return "structured_lookup" if state["route"] == "structured" else "retrieve"

    graph.add_conditional_edges(
        "route_question", route_branch, {"structured_lookup": "structured_lookup", "retrieve": "retrieve"}
    )
    graph.add_conditional_edges(
        "structured_lookup",
        decide_after_structured_lookup,
        {"generate": "generate", "transform_query": "transform_query", "clarify": END},
    )
    graph.add_conditional_edges(
        "retrieve",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    # Always fall back to vectorstore search on retry: a failed structured
    # lookup (e.g. lore/flavor-text questions misrouted to "structured") has
    # no reason to repeat the same query against servants.db, since
    # state["route"] never changes between attempts.
    graph.add_edge("transform_query", "retrieve")

    graph.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "not supported": "increment_generate_retries",
            "not useful": "transform_query",
        },
    )
    graph.add_edge("increment_generate_retries", "generate")

    return graph.compile()


def run_single_hop(question: str) -> dict:
    subgraph = build_subgraph()
    initial_state: SubState = {
        "question": question,
        "route": "",
        "servant_name": "",
        "class_hint": "",
        "documents": [],
        "generation": "",
        "retrieve_retries": 0,
        "generate_retries": 0,
        "needs_clarification": False,
        "clarification_question": "",
    }
    return subgraph.invoke(initial_state)
