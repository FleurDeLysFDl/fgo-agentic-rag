"""Streamlit demo for the FGO Agentic RAG pipeline.

Usage:
    streamlit run app.py

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
    "混合检索（BM25+bge-m3+RRF+重排）+ 结构化数据库查询 + LangGraph Self-RAG 单跳/多跳问答"
)

question = st.text_input(
    "输入关于FGO从者的问题", placeholder="例如：阿尔托莉雅和贞德的宝具阶级哪个更高？"
)

if st.button("提问", type="primary") and question.strip():
    with st.spinner("检索并生成回答中..."):
        result = answer(question)

    st.subheader("最终回答")
    st.write(result["final_answer"])

    if len(result["sub_questions"]) > 1:
        st.subheader("问题拆解")
        for i, sub_q in enumerate(result["sub_questions"], 1):
            st.markdown(f"**子问题 {i}：** {sub_q}")

    st.subheader("子问题回答与引用来源")
    for i, (sub_q, sub_a, docs) in enumerate(
        zip(result["sub_questions"], result["sub_answers"], result["sub_documents"]), 1
    ):
        with st.expander(f"{i}. {sub_q}"):
            st.markdown("**回答：**")
            st.write(sub_a)
            st.markdown("**检索到的资料：**")
            if docs:
                for doc in docs:
                    st.markdown(f"- 来源：*{doc['source']}*")
                    st.text(doc["text"][:300] + ("..." if len(doc["text"]) > 300 else ""))
            else:
                st.markdown("_（未检索到相关资料）_")
