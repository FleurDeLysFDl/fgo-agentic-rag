"""Shared loader: turns data/quest_raw/manifest.jsonl + the cached raw script
files under data/quest_raw/scripts_txt/ into the list of retrieval records
used by build_bm25_index.py / build_vector_index.py, exactly mirroring
wiki_raw_loader.py's contract (one record = one {"text","source","url",
"chunk_id"} dict).

Granularity choice: unlike wiki_raw_loader (one record per SERVANT, everything
merged), here it's one record per QUEST -- a quest/valentine-episode is
already a naturally-bounded narrative unit (a few KB to a few dozen KB), and
main-story quests aren't "about" any single servant, so there's no sensible
per-servant merge target for them. No further chunking is applied within a
quest, consistent with the project's overall "don't chunk, unmodified content
units" decision.

Entries whose script files haven't been downloaded yet (cache miss) are
skipped rather than erroring, so this loader works correctly whether
fetch_quest_scripts.py has fully finished or not.

Also loads data/quest_raw/manifest_jp.jsonl (if present) -- JP-exclusive story
content not yet ported to CN (see fetch_quest_scripts.py --jp-exclusive-only),
reading its scripts from scripts_txt_jp/ instead. Those records carry
"language": "ja" (everything else defaults to "zh") so consumers can tell
untranslated Japanese text apart from the rest of the corpus.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QUEST_RAW_DIR, QUEST_SCRIPTS_DIR, QUEST_SCRIPTS_DIR_JP
from quest_script_parser import render_script

MANIFEST_PATH = QUEST_RAW_DIR / "manifest.jsonl"
MANIFEST_PATH_JP = QUEST_RAW_DIR / "manifest_jp.jsonl"


def build_source(entry: dict) -> str:
    parts = []
    if entry.get("war_long_name"):
        parts.append(entry["war_long_name"])
    if entry.get("spot_name"):
        parts.append(entry["spot_name"])
    if entry.get("name"):
        parts.append(entry["name"])
    source = "／".join(parts) if parts else entry["quest_id"]

    linked = entry.get("linked_servants") or []
    if linked:
        source = f"{'/'.join(linked)}｜{source}"
    return source


def build_url(entry: dict, region: str = "CN") -> str:
    qid = entry["quest_id"]
    if qid.isdigit():
        return f"https://apps.atlasacademy.io/db/{region}/quest/{qid}/1"
    # valentine_{svt_id}_{i} and any other non-numeric pseudo-id: fall back
    # to the first script's raw CDN url, there's no dedicated DB page for it.
    for phase in entry.get("phases", []):
        for sc in phase.get("scripts", []):
            return sc["url"]
    return ""


PHASE_LABEL_BY_LANG = {"zh": "第{n}节", "ja": "第{n}部"}


def build_full_text(entry: dict, scripts_dir: Path, language: str = "zh") -> str:
    phase_blocks = []
    multi_phase = len(entry.get("phases", [])) > 1
    for phase in entry.get("phases", []):
        script_blocks = []
        for sc in phase.get("scripts", []):
            script_path = scripts_dir / f"{sc['script_id']}.txt"
            if not script_path.exists():
                continue
            raw = script_path.read_text(encoding="utf-8")
            rendered = render_script(raw, language)
            if rendered.strip():
                script_blocks.append(rendered.strip())
        if not script_blocks:
            continue
        block = "\n\n".join(script_blocks)
        if multi_phase:
            label = PHASE_LABEL_BY_LANG.get(language, "第{n}节").format(n=phase.get("phase"))
            block = f"【{label}】\n{block}"
        phase_blocks.append(block)

    return "\n\n".join(phase_blocks)


def _load_manifest(manifest_path: Path, scripts_dir: Path, region: str, language: str) -> list[dict]:
    if not manifest_path.exists():
        return []

    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = build_full_text(entry, scripts_dir, language)
            if not text.strip():
                continue  # script(s) not downloaded yet, or genuinely empty
            records.append(
                {
                    "text": text,
                    "source": build_source(entry),
                    "url": build_url(entry, region),
                    "chunk_id": f"quest_{entry['quest_id']}",
                    "language": language,
                }
            )
    return records


def load_records() -> list[dict]:
    records = _load_manifest(MANIFEST_PATH, QUEST_SCRIPTS_DIR, "CN", "zh")
    records += _load_manifest(MANIFEST_PATH_JP, QUEST_SCRIPTS_DIR_JP, "JP", "ja")
    return _dedupe_exact_text(records)


def _dedupe_exact_text(records: list[dict]) -> list[dict]:
    """Collapse quests whose rendered text is byte-identical to another quest's.

    Confirmed corpus-wide (2026-07): ~28% of quest records (1268 of 4507) are
    exact-text duplicates of another record, for three legitimate game-design
    reasons rather than a parsing bug:
      - "archive"/lore pickup quests reachable from several routes/locations
        that all show the identical collected document text (e.g. "纳凉梦幻
        冥峰" event's 档案I-IX, each read from up to 16 different quest_ids).
      - full event reruns ("复刻版") that reuse the original event's script
        verbatim under new quest_ids.
      - Lostbelt "回顾关卡" (recap) stages that replay an earlier chapter's
        dialogue unchanged for players catching up.
    Indexing every copy would waste ~28% of embedding/BM25 work and return
    redundant duplicate hits to the retriever for the same query. Keep one
    representative record per unique text (first-seen in manifest order) and
    track the other quest_ids that shared it in `duplicate_ids`, so provenance
    isn't silently lost even though only one copy is indexed.
    """
    by_text: dict[str, dict] = {}
    for r in records:
        existing = by_text.get(r["text"])
        if existing is None:
            r["duplicate_ids"] = []
            by_text[r["text"]] = r
        else:
            existing["duplicate_ids"].append(r["chunk_id"])
    return list(by_text.values())
