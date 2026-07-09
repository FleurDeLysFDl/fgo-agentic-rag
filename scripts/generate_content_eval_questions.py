"""Generate content-grounded eval questions -- (question, expected_source,
ground_truth) triples produced by having an LLM read each sampled record's
summary-or-full-text (same target build_vector_index.py embeds, see
corpus_text.summary_or_text_for) and write one specific factual question whose
answer can only be found by actually reading it.

This complements generate_bulk_eval_questions.py's template questions, which
just wrap a record's own title ("{title}是谁？") and therefore only test
whether retrieval can match a query against its own title string -- a much
easier task than a real user question that describes a plot point, a
relationship, or a specific detail without naming the record outright. Output
is questions.json-shaped (question/expected_source/ground_truth), so it plugs
into both eval_recall.py (--questions) and eval_ragas.py (--questions) as a
harder recall benchmark and as an answer-quality reference set.

Caching: results are cached to data/content_question_cache.jsonl, keyed by
chunk_id + a hash of whatever text was sent to the LLM, mirroring
summarize_corpus.py's resumable/idempotent design.

Usage:
    python scripts/generate_content_eval_questions.py
    python scripts/generate_content_eval_questions.py --wiki-n 100 --quest-n 100 --seed 42
"""

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR
from corpus_text import load_summary_cache, summary_or_text_for
from wiki_raw_loader import load_records as load_wiki_records
from quest_raw_loader import load_records as load_quest_records
from agent.llm import get_llm

CACHE_PATH = DATA_DIR / "content_question_cache.jsonl"
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions_bulk_content.json"
MAX_RETRIES = 3
WORKERS = 8

PROMPT_TEMPLATE = """你是Fate/Grand Order（FGO）知识问答出题助手。请阅读下面这段游戏文本，出一道具体的事实类问题，并给出唯一确定的参考答案。

要求：
- 问题必须针对文本里的具体情节、人物关系、数值、台词、决定或转折点，答案必须能在文本中明确找到依据。
- 不要出"这是谁""出处是什么""性格是什么样的"这类只凭标题就能猜到答案的泛泛问题。
- 问题里不要直接抄录标题原文（可以自然提到人物/地点名）。
- 参考答案控制在1-2句话，直接给出结论，不要复述问题。
- 严格输出JSON，格式为{{"question": "...", "ground_truth": "..."}}，不要输出其他任何文字。

标题：{source}
正文：
{text}"""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache() -> dict:
    cache = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cache[entry["chunk_id"]] = entry
    return cache


def generate_one(llm, source: str, text: str) -> dict:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = llm.invoke(PROMPT_TEMPLATE.format(source=source, text=text))
            raw = (resp.content or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            question = (data.get("question") or "").strip()
            ground_truth = (data.get("ground_truth") or "").strip()
            if question and ground_truth:
                return {"question": question, "ground_truth": ground_truth}
            last_exc = ValueError(f"empty question/ground_truth: {raw!r}")
        except Exception as exc:  # noqa: BLE001 -- retry any transient API/parse error
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"generate_one failed after {MAX_RETRIES} attempts: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-n", type=int, default=100, help="how many servant-profile questions to sample")
    parser.add_argument("--quest-n", type=int, default=100, help="how many quest/story questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    wiki_records = load_wiki_records()
    quest_records = load_quest_records()
    print(f"corpus: {len(wiki_records)} servant profile(s), {len(quest_records)} quest/story record(s)")

    sample = rng.sample(wiki_records, min(args.wiki_n, len(wiki_records)))
    sample += rng.sample(quest_records, min(args.quest_n, len(quest_records)))

    summary_cache = load_summary_cache()
    cache = load_cache()

    to_process = []
    reused = 0
    for r in sample:
        body = summary_or_text_for(r, summary_cache)
        h = _text_hash(body)
        cached = cache.get(r["chunk_id"])
        if cached and cached.get("text_hash") == h:
            reused += 1
        else:
            to_process.append((r, body, h))

    print(f"{reused} already cached, {len(to_process)} to generate now")

    if to_process:
        llm = get_llm()
        failed = []
        write_lock = threading.Lock()

        def worker(r: dict, body: str, h: str):
            qa = generate_one(llm, r["source"], body)
            return r["chunk_id"], r["source"], h, qa

        with CACHE_PATH.open("a", encoding="utf-8") as out_f:
            with tqdm(total=len(to_process), desc="generating questions", unit="record") as bar:
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    futures = {pool.submit(worker, r, body, h): r["chunk_id"] for r, body, h in to_process}
                    for fut in as_completed(futures):
                        chunk_id = futures[fut]
                        try:
                            cid, source, h, qa = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            failed.append((chunk_id, str(exc)))
                            tqdm.write(f"  [FAIL] {chunk_id}: {exc}")
                            bar.update(1)
                            continue
                        entry = {
                            "chunk_id": cid,
                            "source": source,
                            "text_hash": h,
                            "question": qa["question"],
                            "ground_truth": qa["ground_truth"],
                        }
                        with write_lock:
                            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            out_f.flush()
                        bar.set_postfix(failed=len(failed))
                        bar.update(1)

        print(f"done: {len(to_process) - len(failed)}/{len(to_process)} generated, {len(failed)} failed")
        if failed:
            print("WARNING: failed chunk_ids (re-run this script to retry them):")
            for cid, err in failed[:20]:
                print(f"  {cid}: {err}")

    cache = load_cache()  # reload to pick up entries written just now
    questions = []
    for r in sample:
        entry = cache.get(r["chunk_id"])
        if entry:
            questions.append(
                {
                    "question": entry["question"],
                    "expected_source": entry["source"],
                    "ground_truth": entry["ground_truth"],
                }
            )
    rng.shuffle(questions)

    OUT_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(questions)} question(s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
