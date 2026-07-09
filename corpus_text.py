"""Text used for BM25 tokenization and cross-encoder reranking: a corpus record's
summary-or-full-text, prefixed with its own title (`source`). Dense embedding
deliberately does NOT use this -- it keeps embedding summary_or_text_for() alone,
unprefixed, unchanged from before.

Quest chapter titles and the servant(s) a quest/valentine-script is about often
never appear verbatim in the dialogue body itself, so without the prefix BM25
has no way to match on them at all -- only the rerank stage saw it before
(rerank_text_for), which is too late if the right record never made it into the
top-20 fused candidates in the first place (see eval/recall_results_bulk.log
misses, 2026-07: sibling-chapter confusion within the same event, and
cross-servant confusion on templated valentine scripts).
"""

import hashlib
import json

from config import DATA_DIR

SUMMARY_CACHE_PATH = DATA_DIR / "summary_cache.jsonl"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_summary_cache() -> dict:
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


def summary_or_text_for(record: dict, summary_cache: dict) -> str:
    cached = summary_cache.get(record["chunk_id"])
    if cached and cached.get("text_hash") == _text_hash(record["text"]):
        return cached["summary"]
    return record["text"]


def index_text_for(record: dict, summary_cache: dict) -> str:
    return f"{record['source']}\n{summary_or_text_for(record, summary_cache)}"
