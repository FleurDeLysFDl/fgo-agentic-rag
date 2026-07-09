"""Build a BM25 keyword index over the combined corpus: data/wiki_raw/*.json
(one record per servant -- see wiki_raw_loader.py) plus data/quest_raw/
(one record per quest/valentine script -- see quest_raw_loader.py).

Chinese has no whitespace word boundaries, so rank_bm25 (which expects
pre-tokenized term lists) needs jieba segmentation first -- plain
str.split() would treat each record as one giant "word".

IMPORTANT: the records list order here must exactly match the order used in
build_vector_index.py, since BM25 result indices and Qdrant point ids are
both positional into this same list.

Usage:
    python scripts/build_bm25_index.py
"""

import pickle
import sys
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR
from corpus_text import index_text_for, load_summary_cache
from wiki_raw_loader import load_records as load_wiki_records
from quest_raw_loader import load_records as load_quest_records

BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    return [tok for tok in jieba.cut_for_search(text) if tok.strip()]


def main() -> None:
    wiki_records = load_wiki_records()
    quest_records = load_quest_records()
    records = wiki_records + quest_records

    summary_cache = load_summary_cache()
    tokenized_corpus = [tokenize(index_text_for(r, summary_cache)) for r in records]
    bm25 = BM25Okapi(tokenized_corpus)

    with BM25_INDEX_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "records": records}, f)

    print(
        f"done: BM25 index over {len(records)} records "
        f"({len(wiki_records)} servant + {len(quest_records)} quest) "
        f"written to {BM25_INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
