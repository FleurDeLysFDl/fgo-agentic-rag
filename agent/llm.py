"""Shared LLM client for the LangGraph agent (OpenAI-compatible endpoint)."""

import itertools
import logging
import time
from functools import lru_cache
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


def _preview(text: str, limit: int = 150) -> str:
    text = " ".join(text.split())  # collapse newlines/whitespace for one-line log entries
    return text if len(text) <= limit else text[: limit] + "..."


class LLMCallLogger(BaseCallbackHandler):
    """Fires on every individual chat-completion call made through any
    ChatOpenAI instance built by get_llm() -- covers plain .invoke() calls
    *and* .with_structured_output() calls (structured output just binds a
    tool/parser around the same underlying model, so these callbacks still
    trigger once per actual API round-trip). This gives one log line per LLM
    call regardless of which graph node/subgraph triggered it, instead of
    hand-adding a log statement at every one of the ~9 call sites."""

    def __init__(self) -> None:
        self._pending: dict[UUID, tuple[float, int]] = {}
        self._counter = itertools.count(1)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Note: LangGraph propagates a single ambient run_id through contextvars
        # to every node's plain llm.invoke() call in a given graph.invoke(),
        # so run_id is NOT unique per LLM call here -- use our own counter
        # instead to number calls legibly in the log (start/end still pair up
        # correctly via the dict since calls in this pipeline are sequential,
        # never concurrent).
        call_no = next(self._counter)
        self._pending[run_id] = (time.perf_counter(), call_no)
        last_human = ""
        if messages and messages[0]:
            last_human = messages[0][-1].content if isinstance(messages[0][-1].content, str) else str(messages[0][-1].content)
        logger.info("llm_call #%d: start prompt=%r", call_no, _preview(last_human))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start_time, call_no = self._pending.pop(run_id, (time.perf_counter(), -1))
        elapsed = time.perf_counter() - start_time
        text = ""
        try:
            text = response.generations[0][0].text
        except (IndexError, AttributeError):
            pass
        token_usage = None
        if isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage")
        logger.info(
            "llm_call #%d: end elapsed=%.2fs tokens=%s response=%r",
            call_no,
            elapsed,
            token_usage,
            _preview(text),
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        _, call_no = self._pending.pop(run_id, (0.0, -1))
        logger.error("llm_call #%d: error %s", call_no, error)


_llm_call_logger = LLMCallLogger()


# Without an explicit timeout, ChatOpenAI falls back to the openai SDK's
# default (600s), so a stall on the upstream proxy (observed: single calls
# hanging 2+ minutes with 0% local CPU -- i.e. genuinely stuck waiting on the
# network, not slow inference) blocks the whole graph for up to 10 minutes
# with no feedback. 60s is generous for this pipeline's normal per-call
# latency (typically 1-5s, worst observed non-stalled call ~30s) while still
# failing fast enough to be noticeable. max_retries=2 gives a stalled call a
# couple of fresh attempts rather than one long wait.
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 2


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_API_BASE,
        api_key=LLM_API_KEY,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        callbacks=[_llm_call_logger],
    )
