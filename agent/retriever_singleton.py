"""Lazy singleton for HybridRetriever -- it loads bge-m3 + bge-reranker-v2-m3,
so we only want to pay that cost once per process, not once per graph node call."""

from functools import lru_cache

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval import HybridRetriever


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
