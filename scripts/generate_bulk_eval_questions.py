"""Auto-generate a large (question, expected_source) sample set for Recall@5
evaluation (scripts/eval_recall.py), complementing the 25 small hand-written/
verified questions in eval/questions.json with much broader statistical
coverage across the corpus.

Unlike the hand-written set (where ground_truth answer text was manually
checked against source material), these are template-generated and only
carry expected_source -- reliable by construction (the question always names
the record's own already-disambiguated source title, e.g. "阿尔托莉雅·潘德
拉贡〔Alter〕(Lancer)的出处是什么？"), no manual verification needed, which
is what makes generating hundreds of them at once tractable. They test pure
retrieval (can the hybrid retriever find record X given a query that
mentions X's exact title), not answer-quality -- ground_truth is
deliberately omitted since these aren't meant for the RAGAS answer-quality
harness (eval_ragas.py).

Usage:
    python scripts/generate_bulk_eval_questions.py
    python scripts/generate_bulk_eval_questions.py --wiki-n 150 --quest-n 150 --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quest_raw_loader import load_records as load_quest_records
from wiki_raw_loader import load_records as load_wiki_records

OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions_bulk.json"

WIKI_TEMPLATES = [
    "{title}是谁？",
    "{title}的出处是什么？",
    "{title}的性格是什么样的？",
    "{title}有什么样的经历？",
]

QUEST_TEMPLATES = [
    "「{title}」这段剧情讲的是什么内容？",
    "在「{title}」中发生了什么？",
]


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

    wiki_sample = rng.sample(wiki_records, min(args.wiki_n, len(wiki_records)))
    quest_sample = rng.sample(quest_records, min(args.quest_n, len(quest_records)))

    questions = []
    for r in wiki_sample:
        template = rng.choice(WIKI_TEMPLATES)
        questions.append({"question": template.format(title=r["source"]), "expected_source": r["source"]})
    for r in quest_sample:
        template = rng.choice(QUEST_TEMPLATES)
        questions.append({"question": template.format(title=r["source"]), "expected_source": r["source"]})

    rng.shuffle(questions)

    OUT_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(questions)} question(s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
