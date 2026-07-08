"""Hybrid retrieval pipeline: BM25 + dense (bge-m3) -> RRF fusion -> top-20
-> bge-reranker-v2-m3 cross-encoder rerank -> top-5.

Usage (CLI):
    python retrieval.py "阿尔托莉雅的宝具是什么"

Usage (library):
    from retrieval import HybridRetriever
    r = HybridRetriever()
    r.query("阿尔托莉雅的宝具是什么", top_k=5)
"""

import argparse
import hashlib
import json
import pickle

import jieba
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    RERANKER_MODEL_NAME,
)

BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
SUMMARY_CACHE_PATH = DATA_DIR / "summary_cache.jsonl"
RRF_K = 60  # standard RRF damping constant
DENSE_TOP_K = 20
BM25_TOP_K = 20
FUSED_TOP_K = 20  # candidates sent into the reranker
# Without an explicit cap, CrossEncoder falls back to the tokenizer's
# model_max_length (8192 for bge-reranker-v2-m3). Candidate chunks can run
# up to ~37k chars (full quest scripts), so uncapped reranking was tokenizing
# to 8192 tokens per pair -- 256x the attention cost of a 512-token pair --
# which made every query take minutes instead of seconds.
RERANKER_MAX_LENGTH = 512


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_summary_cache() -> dict:
    """Same cache scripts/summarize_corpus.py writes and build_vector_index.py
    embeds -- reused here so long candidates get reranked against their
    entity-dense summary instead of a truncated slice of raw text."""
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


class HybridRetriever:
    def __init__(self):
        with BM25_INDEX_PATH.open("rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.records = bm25_data["records"]  # index-aligned with BM25 corpus and Qdrant point ids
        self.summary_cache = load_summary_cache()

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.qdrant = QdrantClient(path=str(QDRANT_PATH))
        self._reranker = None  # lazy-loaded, only needed for query()

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH)
        return self._reranker

    def dense_search(self, question: str, top_k: int = DENSE_TOP_K) -> list[tuple[int, float]]:
        vector = self.embed_model.encode(question, normalize_embeddings=True).tolist()
        hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION, query=vector, limit=top_k
        ).points
        return [(hit.id, hit.score) for hit in hits]

    def bm25_search(self, question: str, top_k: int = BM25_TOP_K) -> list[tuple[int, float]]:
        tokens = [tok for tok in jieba.cut_for_search(question) if tok.strip()]
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(idx, scores[idx]) for idx in ranked]

    def enumerate_by_keyword(self, keyword: str) -> list[dict]:
        """Exhaustive substring scan across every record's source/text --
        NOT a top-K similarity search. For "list every quest/chunk mentioning
        X" questions, semantic/BM25 top-K retrieval only surfaces whichever
        handful of records best match the query's specific phrasing and
        silently misses most genuine occurrences (observed: two rephrasings
        of "list every quest 伊阿宋 appears in" returned 6 and 2 records with
        zero overlap, out of an unknown true total). A plain substring check
        over ~4000 records is cheap enough to just do exhaustively instead."""
        return [r for r in self.records if keyword in r["source"] or keyword in r["text"]]

    def rerank_text_for(self, record: dict) -> str:
        """Text fed to the cross-encoder for a candidate. Long records (the
        ones that would get silently truncated at RERANKER_MAX_LENGTH, losing
        whatever content falls past the cutoff) use their cached LLM summary
        instead -- same summary already used for dense embedding, prefixed
        with the title/source so the reranker sees which entity/chapter it's
        looking at. Short records use the full text as-is (nothing to gain
        from summarizing, and nothing gets truncated)."""
        cached = self.summary_cache.get(record["chunk_id"])
        if cached and cached.get("text_hash") == _text_hash(record["text"]):
            return f"{record['source']}\n{cached['summary']}"
        return record["text"]

    @staticmethod
    def rrf_fuse(*rankings: list[tuple[int, float]], k: int = RRF_K) -> list[int]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across each
        ranking list, rank being 1-indexed position. Robust to the very
        different score scales of BM25 vs cosine similarity."""
        fused_scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _score) in enumerate(ranking, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        dense_hits = self.dense_search(question)
        bm25_hits = self.bm25_search(question)
        fused_ids = self.rrf_fuse(dense_hits, bm25_hits)[:FUSED_TOP_K]

        candidates = [self.records[doc_id] for doc_id in fused_ids]
        pairs = [[question, self.rerank_text_for(c)] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [{**chunk, "rerank_score": float(score)} for chunk, score in ranked[:top_k]]

    def query_verbose(self, question: str, top_k: int = 5) -> dict:
        """Same pipeline as query(), but also returns every intermediate
        stage (dense hits, bm25 hits, fused ranking, rerank scores) so
        callers can print/inspect each step."""
        dense_hits = self.dense_search(question)
        bm25_hits = self.bm25_search(question)
        fused_ids = self.rrf_fuse(dense_hits, bm25_hits)[:FUSED_TOP_K]

        candidates = [self.records[doc_id] for doc_id in fused_ids]
        pairs = [[question, self.rerank_text_for(c)] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        final = [{**chunk, "rerank_score": float(score)} for chunk, score in ranked[:top_k]]

        return {
            "dense_hits": [
                {"source": self.records[doc_id]["source"], "score": score}
                for doc_id, score in dense_hits
            ],
            "bm25_hits": [
                {"source": self.records[doc_id]["source"], "score": score}
                for doc_id, score in bm25_hits
            ],
            "fused": [self.records[doc_id]["source"] for doc_id in fused_ids],
            "reranked": [
                {"source": chunk["source"], "score": float(score)}
                for chunk, score in ranked
            ],
            "final": final,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="query text")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.query(args.question, top_k=args.top_k)
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] score={r['rerank_score']:.4f} source={r['source']} ({r['chunk_id']})")
        print(r["text"])


if __name__ == "__main__":
    main()
