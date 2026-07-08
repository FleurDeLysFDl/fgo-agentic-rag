"""FastAPI wrapper over the agentic RAG pipeline.

Each request carries a session_id that addresses its own conversation
memory (agent/memory.py's ConversationMemory, SQLite-backed) -- unlike
app.py, which hardcodes a single "default" session for local interactive
use, this lets arbitrary callers maintain independent, persistent
conversations by supplying whatever id they like (a user id, a chat-widget
session token, etc.). clarification_streak is read from persisted
per-turn state (not an in-process list), so it's correct even if a
follow-up request lands on a different worker process than the one that
asked the clarifying question.

Usage:
    uvicorn api:app --reload
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \\
        -d '{"session_id": "user-42", "message": "阿尔托莉雅的宝具是什么？"}'
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

LOG_PATH = Path(__file__).resolve().parent / ".tmp" / "api.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
)
logging.getLogger("agent").setLevel(logging.INFO)

from agent.graph import answer
from agent.memory import ConversationMemory

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FGO Agentic RAG API",
    description="Hybrid retrieval + LangGraph self-correction + per-session persistent memory.",
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    needs_clarification: bool
    sub_questions: list[str] = []
    sources: list[str] = []


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Plain `def`, not `async def`: agent.graph.answer() is synchronous/
    blocking (model inference, SQLite, network calls), and FastAPI runs
    sync endpoint functions in a thread pool automatically -- declaring
    this `async def` while calling blocking code directly would instead
    block the whole event loop for the entire request."""
    memory = ConversationMemory(session_id=req.session_id)
    streak = memory.clarification_streak()
    history_summary, recent_turns = memory.get_context()

    memory.append_turn("user", req.message)
    logger.info("chat: session_id=%r message=%r clarification_rounds=%d", req.session_id, req.message, streak)

    result = answer(
        req.message,
        history_summary=history_summary,
        recent_turns=recent_turns,
        clarification_rounds=streak,
    )

    memory.append_turn(
        "assistant", result["final_answer"], is_clarification=result["needs_clarification"]
    )

    sources = sorted({doc["source"] for docs in result.get("sub_documents", []) for doc in docs})
    return ChatResponse(
        answer=result["final_answer"],
        needs_clarification=result["needs_clarification"],
        sub_questions=result.get("sub_questions", []),
        sources=sources,
    )


@app.get("/chat/{session_id}/history")
def get_history(session_id: str) -> list[dict]:
    return ConversationMemory(session_id=session_id).load_history()


@app.delete("/chat/{session_id}")
def clear_session(session_id: str) -> dict:
    ConversationMemory(session_id=session_id).clear()
    return {"cleared": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
