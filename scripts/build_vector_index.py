"""Embed the combined corpus -- data/wiki_raw/*.json (one record per servant,
see wiki_raw_loader.py) plus data/quest_raw/ (one record per quest/valentine
script, see quest_raw_loader.py) -- with bge-m3 and index it into a local
(embedded, no server/Docker needed) Qdrant collection.

IMPORTANT: the records list order here must exactly match the order used in
build_bm25_index.py, since Qdrant point ids and BM25 result indices are both
positional into this same list.

Summarization: wiki/quest records are stored as ONE unchunked unit each (see
their loaders' docstrings), which for long records (up to ~30-37K chars)
dilutes a single dense-embedding vector badly. scripts/summarize_corpus.py
pre-generates a ~150-300 char LLM summary for every record over
SUMMARY_THRESHOLD_CHARS and caches it to data/summary_cache.jsonl keyed by
chunk_id + a hash of the source text. Here, if a valid (hash-matching) cached
summary exists for a record, THAT is what gets embedded instead of the full
text -- but the Qdrant payload always stores the full original `text`, so
citation/answer-synthesis is unaffected; only the embedding target changes.
Records with no cached summary (short records, or summarize_corpus.py not run
yet) fall back to embedding the full text, exactly as before.

Resumability: unlike the original version of this script (which always
deleted and re-embedded the whole collection from scratch -- painful on a
~4000-record corpus if the run gets interrupted or hangs partway through),
each point's payload carries an `embed_hash` = sha256 of whatever text was
actually embedded for it (summary or full text). On every run, existing
points are checked by id and re-embedded/upserted ONLY if their stored
embed_hash doesn't match what would be embedded now (new record, or text/
summary changed) -- everything else is skipped. This mirrors the skip-if-
cached convention used elsewhere in this project (scrape_wiki.py,
fetch_quest_scripts.py). If the record count shrinks (e.g. a quest gets
deduped away), leftover trailing point ids beyond the current record count
are deleted so the collection doesn't accumulate stale orphans.

Use --force to bypass all of this and rebuild the collection from scratch
(delete + re-embed everything), e.g. after a change to EMBEDDING_MODEL_NAME.

Usage:
    python scripts/build_vector_index.py
    python scripts/build_vector_index.py --force
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, EMBEDDING_MODEL_NAME, QDRANT_COLLECTION, QDRANT_PATH
from wiki_raw_loader import load_records as load_wiki_records
from quest_raw_loader import load_records as load_quest_records

BATCH_SIZE = 32
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


def embedding_text_for(record: dict, summary_cache: dict) -> str:
    cached = summary_cache.get(record["chunk_id"])
    if cached and cached.get("text_hash") == _text_hash(record["text"]):
        return cached["summary"]
    return record["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-embed the whole collection from scratch instead of "
        "skipping records whose embed_hash is already up to date.",
    )
    args = parser.parse_args()

    wiki_records = load_wiki_records()
    quest_records = load_quest_records()
    records = wiki_records + quest_records
    print(
        f"loaded {len(records)} records "
        f"({len(wiki_records)} servant + {len(quest_records)} quest)"
    )

    summary_cache = load_summary_cache()
    used_summary = sum(
        1 for r in records
        if summary_cache.get(r["chunk_id"], {}).get("text_hash") == _text_hash(r["text"])
    )
    print(
        f"summary cache: {len(summary_cache)} entries loaded, "
        f"{used_summary}/{len(records)} records will embed their summary "
        f"(rest embed full text)"
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(path=str(QDRANT_PATH))
    existed = client.collection_exists(QDRANT_COLLECTION)
    if existed and args.force:
        client.delete_collection(QDRANT_COLLECTION)
        existed = False
    if not existed:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    old_count = client.get_collection(QDRANT_COLLECTION).points_count if existed else 0
    if existed:
        print(f"collection already exists with {old_count} points -- resuming incrementally (skip up-to-date records)")

    embedded = 0
    skipped = 0
    with tqdm(total=len(records), desc="embedding + indexing", unit="record") as bar:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            ids = list(range(start, start + len(batch)))
            target_texts = [embedding_text_for(r, summary_cache) for r in batch]
            target_hashes = [_text_hash(t) for t in target_texts]

            existing_hashes: dict[int, str | None] = {}
            if existed:
                fetched = client.retrieve(
                    collection_name=QDRANT_COLLECTION, ids=ids, with_payload=["embed_hash"]
                )
                existing_hashes = {p.id: (p.payload or {}).get("embed_hash") for p in fetched}

            need_idx = [i for i, pid in enumerate(ids) if existing_hashes.get(pid) != target_hashes[i]]
            skipped += len(batch) - len(need_idx)

            if need_idx:
                texts_to_encode = [target_texts[i] for i in need_idx]
                embeddings = model.encode(
                    texts_to_encode, normalize_embeddings=True, show_progress_bar=False
                )
                points = []
                for j, i in enumerate(need_idx):
                    payload = dict(batch[i])
                    payload["embed_hash"] = target_hashes[i]
                    points.append(PointStruct(id=ids[i], vector=embeddings[j].tolist(), payload=payload))
                client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                embedded += len(need_idx)

            bar.set_postfix(embedded=embedded, skipped=skipped)
            bar.update(len(batch))

    if old_count > len(records):
        stale_ids = list(range(len(records), old_count))
        client.delete(collection_name=QDRANT_COLLECTION, points_selector=stale_ids)
        print(f"removed {len(stale_ids)} stale trailing points (record count shrank from {old_count} to {len(records)})")

    client.close()
    print(
        f"done: {embedded} embedded, {skipped} already up to date (skipped), "
        f"{len(records)} total records indexed into {QDRANT_PATH} (collection={QDRANT_COLLECTION})"
    )


if __name__ == "__main__":
    main()
