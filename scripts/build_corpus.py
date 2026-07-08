"""Chunk scraped wiki text (profile / April Fools alt-profile / voice lines)
into data/corpus/*.jsonl for retrieval.

Usage:
    python scripts/build_corpus.py

Reads every data/wiki_raw/{svt_id}.json (written by scrape_wiki.py) and
writes to data/corpus/servants.jsonl (one line per chunk, fields: text,
source, url, chunk_id, content_type):
  - profile_parts: paragraph-based prose, packed 300-500 chars (chunk_paragraphs)
    -- this is the one content type that benefits from size-based packing,
    since raw paragraphs vary a lot in length.
  - april_fools_parts: NOT chunked -- joined into a single whole-servant
    retrieval unit (it's already short, a couple of paragraphs at most).
  - voice_lines: NOT chunked -- all sections/lines for a servant are joined
    into a single whole-servant retrieval unit (one chunk per servant, with
    "==Section==" markers kept inline for readability), rather than packed/
    split by size. These are short discrete quotes, not prose meant to be
    read as a size-bounded window -- splitting them up would break the
    natural per-servant grouping for no benefit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CORPUS_DIR, WIKI_RAW_DIR

MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 500


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """Greedily pack paragraphs into chunks within [MIN_CHUNK_CHARS, MAX_CHUNK_CHARS],
    splitting any single paragraph that alone exceeds MAX_CHUNK_CHARS on sentence
    boundaries (中文句号/问号/叹号)."""
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > MAX_CHUNK_CHARS:
            flush()
            sentences = []
            buf = ""
            for ch in para:
                buf += ch
                if ch in "。！？":
                    sentences.append(buf)
                    buf = ""
            if buf:
                sentences.append(buf)
            sub = ""
            for sent in sentences:
                if sub and len(sub) + len(sent) > MAX_CHUNK_CHARS:
                    chunks.append(sub)
                    sub = ""
                sub += sent
            if sub:
                chunks.append(sub)
            continue

        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            flush()
        current = f"{current}\n{para}" if current else para
        if len(current) >= MIN_CHUNK_CHARS:
            flush()

    flush()
    return chunks


def format_voice_line(line: dict) -> str:
    label = line.get("label") or ""
    text = line.get("text") or ""
    condition = line.get("condition") or ""
    if condition:
        return f"{label}（{condition}）：{text}"
    return f"{label}：{text}"


def main() -> None:
    out_path = CORPUS_DIR / "servants.jsonl"
    n_chunks = n_servants = 0
    n_profile = n_april_fools = n_voice = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for raw_path in sorted(WIKI_RAW_DIR.glob("*.json")):
            record = json.loads(raw_path.read_text(encoding="utf-8"))
            svt_id = record["svt_id"]
            resolved_title = record["resolved_title"]
            url = record["url"]
            wrote_any = False

            def emit(chunk: str, source: str, chunk_id: str, content_type: str) -> None:
                out_f.write(
                    json.dumps(
                        {
                            "text": chunk,
                            "source": source,
                            "url": url,
                            "chunk_id": chunk_id,
                            "content_type": content_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            profile_parts = record.get("profile_parts") or []
            for i, chunk in enumerate(chunk_paragraphs(profile_parts)):
                emit(chunk, resolved_title, f"{svt_id}_{i}", "profile")
                n_chunks += 1
                n_profile += 1
                wrote_any = True

            april_fools_parts = record.get("april_fools_parts") or []
            if april_fools_parts:
                text = "\n\n".join(p.strip() for p in april_fools_parts if p.strip())
                if text:
                    emit(text, f"{resolved_title}（愚人节资料）", f"{svt_id}_af", "april_fools")
                    n_chunks += 1
                    n_april_fools += 1
                    wrote_any = True

            voice_sections = record.get("voice_lines") or []
            if voice_sections:
                blocks = []
                for section in voice_sections:
                    title = section.get("title") or "语音"
                    formatted_lines = [format_voice_line(l) for l in section.get("lines", [])]
                    if formatted_lines:
                        blocks.append(f"=={title}==\n" + "\n".join(formatted_lines))
                text = "\n\n".join(blocks)
                if text:
                    emit(text, f"{resolved_title}/语音", f"{svt_id}_voice", "voice_line")
                    n_chunks += 1
                    n_voice += 1
                    wrote_any = True

            if wrote_any:
                n_servants += 1

    print(
        f"wrote {out_path}: {n_chunks} chunks from {n_servants} servants "
        f"(profile={n_profile} april_fools={n_april_fools} voice_line={n_voice})"
    )


if __name__ == "__main__":
    main()
