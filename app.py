"""Streamlit chat demo for the FGO Agentic RAG pipeline.

Usage:
    streamlit run app.py

Multi-turn: each question is sent along with the prior conversation
(st.session_state.history) so the agent can resolve references like "她的
宝具是什么" against whatever servant was just discussed (agent.graph.
resolve_question). If a question is ambiguous even with that history -- or a
structured lookup matches more than one servant variant -- the agent asks a
clarifying question instead of guessing; that comes back as a normal
assistant message (result["needs_clarification"]), so just answering it in
the next turn continues the same conversation.

Pipeline trace (routing decisions, retrieval hit counts, grading judgments,
retries, timings) is printed to the console/terminal running `streamlit run`
via the `agent` logger -- check that terminal (or `preview_logs` if launched
through the preview tool) to see what happened during a query.
"""

import logging

import streamlit as st

# Keep third-party libraries quiet (httpx/urllib3/sentence_transformers etc.
# default to WARNING via basicConfig) but surface our own agent.* pipeline
# trace at INFO so routing/retrieval/grading decisions are visible per query.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("agent").setLevel(logging.INFO)

from agent.graph import answer

st.set_page_config(page_title="FGO Agentic RAG", page_icon="⚔️")
st.title("⚔️ FGO Agentic RAG")
st.caption(
    "混合检索（BM25+bge-m3+RRF+重排）+ 结构化数据库查询 + LangGraph Self-RAG 多轮问答"
    "（带上下文记忆；信息不足或从者形态有歧义时会反问）"
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str}
if "turn_details" not in st.session_state:
    st.session_state.turn_details = []  # parallel to history; None except for non-clarification assistant turns

if st.button("清空对话"):
    st.session_state.history = []
    st.session_state.turn_details = []
    st.rerun()


def render_details(details: dict) -> None:
    if len(details["sub_questions"]) > 1:
        st.markdown("**问题拆解：**")
        for j, sub_q in enumerate(details["sub_questions"], 1):
            st.markdown(f"{j}. {sub_q}")
    with st.expander("子问题回答与引用来源"):
        for sub_q, sub_a, docs in zip(
            details["sub_questions"], details["sub_answers"], details["sub_documents"]
        ):
            st.markdown(f"**{sub_q}**")
            st.write(sub_a)
            if docs:
                for doc in docs:
                    st.markdown(f"- 来源：*{doc['source']}*")
                    st.text(doc["text"][:300] + ("..." if len(doc["text"]) > 300 else ""))
            else:
                st.markdown("_（未检索到相关资料）_")


for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        details = st.session_state.turn_details[i]
        if details:
            render_details(details)

question = st.chat_input("输入关于FGO从者的问题，可以是追问（例如“她的宝具是什么”）")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.turn_details.append(None)
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("检索并生成回答中..."):
            # history excludes the question just appended above -- it's the
            # prior turns the agent uses to resolve references in `question`.
            result = answer(question, history=st.session_state.history[:-1])
        st.write(result["final_answer"])

        details = None
        if not result["needs_clarification"]:
            details = {
                "sub_questions": result["sub_questions"],
                "sub_answers": result["sub_answers"],
                "sub_documents": result["sub_documents"],
            }
            render_details(details)

    st.session_state.history.append({"role": "assistant", "content": result["final_answer"]})
    st.session_state.turn_details.append(details)
