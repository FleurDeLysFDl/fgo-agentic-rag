"""Single-hop Self-RAG-style subgraph: route -> retrieve/structured-lookup ->
grade documents -> (retry with rewritten query if nothing relevant) -> generate
-> grade generation (hallucination + answer-quality) -> retry generate or
retry retrieval if needed, else return.

gpt-4o-mini has not been fine-tuned with literal Self-RAG reflection tokens,
so each "reflection" judgment (retrieve-worthy? relevant? supported? useful?)
is elicited via structured LLM output (agent/schemas.py) instead of special
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
    GradeDocument,
    GradeHallucination,
    RewrittenQuery,
    RouteQuery,
)
from agent.state import SubState
from agent.structured_lookup import lookup_servant

logger = logging.getLogger(__name__)

MAX_RETRIEVE_RETRIES = 2
MAX_GENERATE_RETRIES = 2


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
    documents = [_format_structured_doc(m) for m in matches]
    logger.info(
        "structured_lookup: servant_name=%r class_hint=%r -> %d match(es)",
        state["servant_name"],
        state.get("class_hint"),
        len(matches),
    )
    return {"documents": documents}


def retrieve(state: SubState) -> dict:
    retriever = get_retriever()
    results = retriever.query(state["question"], top_k=5)
    documents = [{"text": r["text"], "source": r["source"]} for r in results]
    logger.info(
        "retrieve: query=%r -> %d result(s): %s",
        state["question"],
        len(documents),
        [d["source"] for d in documents],
    )
    return {"documents": documents}


def grade_documents(state: SubState) -> dict:
    llm = get_llm().with_structured_output(GradeDocument)
    relevant = []
    for doc in state["documents"]:
        result: GradeDocument = llm.invoke(
            [
                ("system", "判断以下资料是否与问题相关，只回答yes或no。"),
                ("human", f"问题：{state['question']}\n\n资料：{doc['text']}"),
            ]
        )
        if result.binary_score == "yes":
            relevant.append(doc)
    logger.info(
        "grade_documents: %d/%d document(s) judged relevant (kept: %s)",
        len(relevant),
        len(state["documents"]),
        [d["source"] for d in relevant],
    )
    return {"documents": relevant}


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
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)
    graph.add_node("increment_generate_retries", increment_generate_retries)

    graph.set_entry_point("route_question")

    def route_branch(state: SubState) -> str:
        return "structured_lookup" if state["route"] == "structured" else "retrieve"

    graph.add_conditional_edges(
        "route_question", route_branch, {"structured_lookup": "structured_lookup", "retrieve": "retrieve"}
    )
    graph.add_edge("structured_lookup", "grade_documents")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
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
    }
    return subgraph.invoke(initial_state)
