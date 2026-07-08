"""Generate LLM summaries for long corpus records, to be used as the DENSE-
EMBEDDING text in build_vector_index.py (BM25 keeps indexing full text
unchanged; the final citation/answer-synthesis text is also always the full
original text -- summaries only ever replace what gets embedded).

Why this exists: both wiki_raw_loader.py and quest_raw_loader.py deliberately
store one record per servant / per quest with NO sub-chunking (explicit
project design choice -- see their docstrings). That's great for BM25 and for
citation, but a single dense-embedding vector for an 8,000-30,000 character
document is known to dilute badly -- bge-m3 can technically encode up to
8192 tokens, but semantically a paragraph-scale summary is a much sharper
target for cosine-similarity search than the raw full text. Corpus-wide
length audit (2026-07):
    wiki:  n=391  min=1038  median=4633  p90=8919   max=32766
    quest: n=3239 min=10    median=3056  p90=10264  max=37191
So instead of chunking (rejected earlier in this project) or embedding raw
long text as-is, records over SUMMARY_THRESHOLD_CHARS get a ~150-300 char
Chinese LLM summary that captures key entities/events/decisions, and THAT is
what gets embedded. Short records skip the LLM call entirely (summary =
full text) since there's nothing to gain from summarizing them.

Caching: results are cached to data/summary_cache.jsonl, keyed by chunk_id +
a sha256 of the source text, so:
  - re-running this script is idempotent/resumable (crash-safe: every
    successful call is appended and flushed immediately, not batched)
  - if a record's underlying text changes (e.g. a future parser fix), its
    hash changes and it's automatically regenerated rather than silently
    reusing a stale summary.

Usage:
    python scripts/summarize_corpus.py
"""

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR
from wiki_raw_loader import load_records as load_wiki_records
from quest_raw_loader import load_records as load_quest_records
from agent.llm import get_llm

SUMMARY_CACHE_PATH = DATA_DIR / "summary_cache.jsonl"
SUMMARY_THRESHOLD_CHARS = 1500
MAX_RETRIES = 3
# LLM calls are I/O-bound (network round-trip to the proxy endpoint), so a
# thread pool speeds this up a lot -- ~2,663 sequential calls would take
# 2+ hours one at a time; with concurrency it's more like 15-20 minutes.
# get_llm() returns a cached ChatOpenAI singleton (functools.lru_cache), and
# LangChain's ChatOpenAI/openai client is documented thread-safe for
# concurrent .invoke() calls from multiple threads.
SUMMARY_WORKERS = 8

PROMPT_TEMPLATE = """你是Fate/Grand Order（FGO）剧情摘要助手。请为下面这段游戏文本生成一段150-300字的中文摘要。

要求：
- 保留关键人物名、地点名、专有名词、关键事件与关键决定/转折点。
- 去除重复的对话细节、台词原文、拟声词等对检索无帮助的内容。
- 摘要将被用作语义向量检索的索引文本，因此需要包含足够的实体和主题信息，
  以便与该内容相关的用户提问能够匹配到这段摘要。
- 直接输出摘要正文，不要加"摘要："等前缀，不要分点。

正文：
{text}"""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    cache = {}
    if SUMMARY_CACHE_PATH.exists():
        with SUMMARY_CACHE_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cache[entry["chunk_id"]] = entry
    return cache


def summarize_one(llm, text: str) -> str:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = llm.invoke(PROMPT_TEMPLATE.format(text=text))
            summary = (resp.content or "").strip()
            if summary:
                return summary
            last_exc = ValueError("empty response")
        except Exception as exc:  # noqa: BLE001 -- retry any transient API error
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"summarize_one failed after {MAX_RETRIES} attempts: {last_exc}")


def main() -> None:
    wiki_records = load_wiki_records()
    quest_records = load_quest_records()
    records = wiki_records + quest_records
    print(
        f"loaded {len(records)} records "
        f"({len(wiki_records)} servant + {len(quest_records)} quest)",
        flush=True,
    )

    cache = load_cache()
    to_process = []
    reused = 0
    short_skipped = 0
    for r in records:
        if len(r["text"]) <= SUMMARY_THRESHOLD_CHARS:
            short_skipped += 1
            continue
        h = _text_hash(r["text"])
        cached = cache.get(r["chunk_id"])
        if cached and cached.get("text_hash") == h:
            reused += 1
            continue
        to_process.append((r, h))

    print(
        f"{short_skipped} records <= {SUMMARY_THRESHOLD_CHARS} chars (no summary needed), "
        f"{reused} already cached, {len(to_process)} to summarize now",
        flush=True,
    )

    if not to_process:
        print("nothing to do", flush=True)
        return

    llm = get_llm()
    failed = []
    write_lock = threading.Lock()

    def worker(r: dict, h: str):
        summary = summarize_one(llm, r["text"])
        return r["chunk_id"], h, len(r["text"]), summary

    with SUMMARY_CACHE_PATH.open("a", encoding="utf-8") as out_f:
        with tqdm(total=len(to_process), desc="summarizing", unit="record") as bar:
            with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as pool:
                futures = {pool.submit(worker, r, h): r["chunk_id"] for r, h in to_process}
                for fut in as_completed(futures):
                    chunk_id = futures[fut]
                    try:
                        cid, h, orig_len, summary = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        failed.append((chunk_id, str(exc)))
                        tqdm.write(f"  [FAIL] {chunk_id}: {exc}")
                        bar.update(1)
                        continue
                    entry = {
                        "chunk_id": cid,
                        "text_hash": h,
                        "summary": summary,
                        "orig_len": orig_len,
                        "summary_len": len(summary),
                    }
                    with write_lock:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        out_f.flush()
                    bar.set_postfix(failed=len(failed))
                    bar.update(1)

    print(f"done: {len(to_process) - len(failed)}/{len(to_process)} summarized, {len(failed)} failed", flush=True)
    if failed:
        print("WARNING: failed chunk_ids (re-run this script to retry them):", flush=True)
        for cid, err in failed[:20]:
            print(f"  {cid}: {err}", flush=True)


if __name__ == "__main__":
    main()
