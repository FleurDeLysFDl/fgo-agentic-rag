"""Shared loader: turns data/wiki_raw/{svt_id}.json files directly into the
list of retrieval records used by both build_bm25_index.py and
build_vector_index.py.

Per explicit design choice: there is no intermediate data/corpus/servants.jsonl
chunking step. Each wiki_raw file becomes exactly ONE record -- its
profile_parts, april_fools_parts and voice_lines are concatenated as-is into
a single text blob, with no size-based packing/splitting (not even the
paragraph-based 300-500 char packing the original profile-only pipeline used
in Phase 0/1). This is a deliberate tradeoff: some servants' combined text
(voice lines especially) can run to ~30k characters in one record, which
trades away the finer retrieval granularity Phase 1's eval (96% recall@5) was
measured against, in exchange for "store each servant's wiki content as one
unit, unmodified in substance."

Both index builders MUST call load_records() (not read the directory
independently) so that BM25 doc-list order and Qdrant point-id order stay
aligned -- see retrieval.py's HybridRetriever, which assumes
self.records[i] and Qdrant point id i refer to the same record.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import WIKI_RAW_DIR


def format_voice_line(line: dict) -> str:
    label = line.get("label") or ""
    text = line.get("text") or ""
    condition = line.get("condition") or ""
    if condition:
        return f"{label}（{condition}）：{text}"
    return f"{label}：{text}"


def build_full_text(record: dict) -> str:
    parts = []

    profile_parts = record.get("profile_parts") or []
    if profile_parts:
        parts.append("\n\n".join(p.strip() for p in profile_parts if p.strip()))

    april_fools_parts = record.get("april_fools_parts") or []
    if april_fools_parts:
        parts.append("【愚人节资料】\n" + "\n\n".join(p.strip() for p in april_fools_parts if p.strip()))

    voice_sections = record.get("voice_lines") or []
    blocks = []
    for section in voice_sections:
        title = section.get("title") or "语音"
        formatted_lines = [format_voice_line(l) for l in section.get("lines", [])]
        if formatted_lines:
            blocks.append(f"=={title}==\n" + "\n".join(formatted_lines))
    if blocks:
        parts.append("【语音台词】\n" + "\n\n".join(blocks))

    return "\n\n".join(parts)


def load_records() -> list[dict]:
    records = []
    for raw_path in sorted(WIKI_RAW_DIR.glob("*.json")):
        record = json.loads(raw_path.read_text(encoding="utf-8"))
        text = build_full_text(record)
        if not text.strip():
            continue  # e.g. NPC/enemy-only pages with no 个人资料 template at all
        records.append(
            {
                "text": text,
                "source": record["resolved_title"],
                "url": record["url"],
                "chunk_id": str(record["svt_id"]),
            }
        )
    return records
