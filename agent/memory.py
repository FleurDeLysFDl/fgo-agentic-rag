"""Bounded, persistent conversation memory.

Two problems with passing app.py's raw session_state history straight into
answer(): (1) it grows without bound -- a long conversation eventually means
stuffing the entire transcript into every resolve_question call, which gets
slow, expensive, and eventually exceeds the model's context window; (2) it
lives only in Streamlit's in-memory session_state, so a page refresh or an
app restart loses the whole conversation.

ConversationMemory fixes both: every turn is persisted to SQLite as it
happens (survives restarts), and get_context() returns a *bounded* view --
the last WINDOW_TURNS turns verbatim, plus a running LLM-generated summary
of everything older than that (recomputed incrementally: only the newly
aged-out turns get folded into the existing summary, not the whole history
every time). That bounded (summary, recent_turns) pair is what actually
flows into the LLM, both for resolve_question and for generate() (see
agent/graph.py, agent/subgraph.py) -- not the unbounded raw transcript.

Usage:
    from agent.memory import ConversationMemory
    memory = ConversationMemory(session_id="default")
    memory.append_turn("user", question)
    summary, recent_turns = memory.get_context()  # excludes the turn just appended
    ...
    memory.append_turn("assistant", final_answer)
"""

import sqlite3
import time

from agent.llm import get_llm
from agent.state import Turn
from config import CONVERSATIONS_DB_PATH

WINDOW_TURNS = 6  # keep the last N raw turns verbatim (~3 user/assistant exchanges)


def format_history(history_summary: str, recent_turns: list[Turn]) -> str:
    """Render a bounded (summary, recent_turns) context -- as returned by
    ConversationMemory.get_context() -- into the text block used in
    prompts. Shared by agent/graph.py's resolve_question and
    agent/subgraph.py's generate() so conversation context is presented the
    same way wherever it's used."""
    parts = []
    if history_summary:
        parts.append(f"（更早的对话摘要）{history_summary}")
    parts.extend(
        f"{'用户' if turn['role'] == 'user' else '助手'}：{turn['content']}" for turn in recent_turns
    )
    return "\n".join(parts)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CONVERSATIONS_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (session_id, turn_index)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            summarized_through INTEGER NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return conn


class ConversationMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    def load_history(self) -> list[Turn]:
        """The full, unabridged transcript in order -- for display purposes
        (the UI should show everything the user actually said), not what
        gets handed to the LLM as context (see get_context)."""
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY turn_index",
                (self.session_id,),
            ).fetchall()
            return [{"role": role, "content": content} for role, content in rows]
        finally:
            conn.close()

    def append_turn(self, role: str, content: str) -> None:
        conn = _connect()
        try:
            next_index = conn.execute(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO turns (session_id, turn_index, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (self.session_id, next_index, role, content, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def clear(self) -> None:
        conn = _connect()
        try:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (self.session_id,))
            conn.commit()
        finally:
            conn.close()

    def get_context(self) -> tuple[str, list[Turn]]:
        """(summary_of_older_turns, recent_verbatim_turns) -- bounded
        regardless of how long the conversation has run. summary is "" if
        the conversation hasn't exceeded WINDOW_TURNS yet."""
        all_turns = self.load_history()
        if len(all_turns) <= WINDOW_TURNS:
            return "", all_turns
        recent = all_turns[-WINDOW_TURNS:]
        older = all_turns[:-WINDOW_TURNS]
        return self._get_or_update_summary(older), recent

    def _get_or_update_summary(self, older_turns: list[Turn]) -> str:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT summary, summarized_through FROM summaries WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            prior_summary, summarized_through = row if row else ("", 0)

            if summarized_through >= len(older_turns):
                return prior_summary  # already covers everything that's aged out of the window

            newly_aged_out = older_turns[summarized_through:]
            new_text = "\n".join(
                f"{'用户' if t['role'] == 'user' else '助手'}：{t['content']}" for t in newly_aged_out
            )
            prompt = (
                (
                    "请在已有摘要的基础上，融合下面新增的对话内容，输出一份更新后的对话摘要"
                    "（保留关键实体、结论、用户已经确认过的选择；简洁，不超过200字，"
                    "只输出摘要正文）。\n\n"
                    f"已有摘要：{prior_summary}\n\n新增对话：\n{new_text}"
                )
                if prior_summary
                else (
                    "请把下面这段对话浓缩成一份简洁摘要（保留关键实体、结论、用户已经确认过的"
                    "选择；不超过200字，只输出摘要正文）。\n\n" + new_text
                )
            )
            new_summary = get_llm().invoke([("human", prompt)]).content.strip()

            conn.execute(
                "INSERT INTO summaries (session_id, summary, summarized_through, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "summary=excluded.summary, summarized_through=excluded.summarized_through, "
                "updated_at=excluded.updated_at",
                (self.session_id, new_summary, len(older_turns), time.time()),
            )
            conn.commit()
            return new_summary
        finally:
            conn.close()
