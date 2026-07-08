"""Streamlit chat demo for the FGO Agentic RAG pipeline.

Usage:
    streamlit run app.py

Multi-turn, with real memory management (agent/memory.py's ConversationMemory)
rather than just an unbounded in-memory list:
  - persisted to SQLite (data/conversations.db), so the conversation survives
    a page refresh or the app process restarting -- not just session_state.
  - bounded: only the last few turns are sent verbatim; anything older is
    folded into a running LLM-generated summary instead of being resent in
    full every turn, so prompt size doesn't grow without bound as the
    conversation gets long.
  - actually used downstream: both resolve_question (to rewrite references
    like "她的宝具是什么" into a self-contained question) and generate() (for
    tone/continuity when producing the answer) see this bounded context, not
    just a single preprocessing step.

If a question is ambiguous even with that context -- or retrieved documents
disagree with each other -- the agent asks a clarifying question instead of
guessing; that comes back as a normal assistant message
(result["needs_clarification"]), so just answering it in the next turn
continues the same conversation.

Pipeline trace (routing decisions, retrieval hit counts, grading judgments,
retries, timings) is printed to the console/terminal running `streamlit run`
via the `agent` logger, AND mirrored to .tmp/streamlit_app.log -- the
terminal is only readable live (nothing to inspect after the fact if the
window isn't watched at the time), so the file exists for post-hoc
debugging of a session from its transcript.
"""

import logging
from pathlib import Path

import streamlit as st

LOG_PATH = Path(__file__).resolve().parent / ".tmp" / "streamlit_app.log"
LOG_PATH.parent.mkdir(exist_ok=True)

# Keep third-party libraries quiet (httpx/urllib3/sentence_transformers etc.
# default to WARNING via basicConfig) but surface our own agent.* pipeline
# trace at INFO so routing/retrieval/grading decisions are visible per query.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)
logging.getLogger("agent").setLevel(logging.INFO)

from agent.graph import answer
from agent.memory import ConversationMemory

# Single-user local app, one persistent conversation thread -- no login/
# multi-tenant session model, so a fixed id is enough to give the whole
# conversation durable identity across restarts.
memory = ConversationMemory(session_id="default")

st.set_page_config(page_title="FGO Agentic RAG", page_icon="⚔️")
st.title("⚔️ FGO Agentic RAG")
st.caption(
    "混合检索（BM25+bge-m3+RRF+重排）+ 结构化数据库查询 + LangGraph Self-RAG 多轮问答"
    "（持久化+有界的对话记忆；信息不足或资料冲突时会反问）"
)

if "turn_details" not in st.session_state:
    # Parallel to memory.load_history(); None except for non-clarification
    # assistant turns. Rebuilt fresh each process start padded to match
    # whatever was already persisted -- turns from a prior run just won't
    # have an expandable "sources" section, only ones added this run will.
    st.session_state.turn_details = [None] * len(memory.load_history())

if st.button("清空对话"):
    memory.clear()
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


history = memory.load_history()
for i, turn in enumerate(history):
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        details = st.session_state.turn_details[i] if i < len(st.session_state.turn_details) else None
        if details:
            render_details(details)

question = st.chat_input("输入关于FGO从者的问题，可以是追问（例如“她的宝具是什么”）")
if question:
    streak = memory.clarification_streak()
    # get_context() BEFORE appending the current question -- it's the prior
    # turns the agent uses to resolve references in `question`, not
    # including `question` itself.
    history_summary, recent_turns = memory.get_context()

    memory.append_turn("user", question)
    st.session_state.turn_details.append(None)
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("检索并生成回答中..."):
            result = answer(
                question,
                history_summary=history_summary,
                recent_turns=recent_turns,
                clarification_rounds=streak,
            )
        st.write(result["final_answer"])

        details = None
        if not result["needs_clarification"]:
            details = {
                "sub_questions": result["sub_questions"],
                "sub_answers": result["sub_answers"],
                "sub_documents": result["sub_documents"],
            }
            render_details(details)

    memory.append_turn(
        "assistant", result["final_answer"], is_clarification=result["needs_clarification"]
    )
    st.session_state.turn_details.append(details)
