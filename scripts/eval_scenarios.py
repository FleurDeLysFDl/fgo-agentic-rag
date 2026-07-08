"""Behavioral eval harness for capabilities eval_recall.py's fact-based
Recall@5 doesn't cover: multi-turn memory, clarification/conflict-detection,
enumerate routing. Instead of comparing generated text to a fixed reference
answer, each scenario checks a BEHAVIOR -- did it ask for clarification? did
it classify this as an enumeration? did a follow-up correctly reuse
conversation memory? -- by inspecting agent.graph.answer()'s full result
dict (needs_clarification, sub_question_plans, final_answer, ...).

Scenarios can be multi-turn: turns run sequentially through the same
ConversationMemory session (mirroring how app.py/api.py drive a real
conversation, see agent/memory.py), each with its own checks.

Flakiness note: several of these behaviors are LLM judgment calls, observed
non-deterministic even at temperature=0 during this project's development
(see docs/BENCHMARKS.md) -- a single run's pass/fail is a sample, not a
guarantee. --repeat N reruns every scenario N times and reports a pass rate.

Usage:
    python scripts/eval_scenarios.py
    python scripts/eval_scenarios.py --repeat 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import answer
from agent.memory import ConversationMemory

SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "eval" / "scenarios.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "scenario_results.log"


def check_turn(result: dict, check: dict) -> tuple[bool, str]:
    if "needs_clarification" in check:
        expected = check["needs_clarification"]
        actual = result["needs_clarification"]
        if actual != expected:
            return False, f"needs_clarification: expected {expected}, got {actual}"

    if "answer_contains_any" in check:
        text = result["final_answer"]
        if not any(s in text for s in check["answer_contains_any"]):
            return False, f"answer missing all of {check['answer_contains_any']!r}: {text[:150]!r}"

    if "answer_not_contains_any" in check:
        text = result["final_answer"]
        hit = [s for s in check["answer_not_contains_any"] if s in text]
        if hit:
            return False, f"answer unexpectedly contains {hit!r}: {text[:150]!r}"

    if "min_sub_questions" in check:
        n = len(result.get("sub_questions", []))
        if n < check["min_sub_questions"]:
            return False, f"sub_questions count {n} < expected minimum {check['min_sub_questions']}"

    if "query_type_any" in check:
        types = [p["query_type"] for p in result.get("sub_question_plans", [])]
        if check["query_type_any"] not in types:
            return False, f"query_type {check['query_type_any']!r} not found among {types!r}"

    if "query_type_all" in check:
        types = [p["query_type"] for p in result.get("sub_question_plans", [])]
        if any(t != check["query_type_all"] for t in types):
            return False, f"expected all query_type={check['query_type_all']!r}, got {types!r}"

    return True, "ok"


def run_scenario(scenario: dict) -> dict:
    session_id = f"eval_scenario_{scenario['id']}"
    memory = ConversationMemory(session_id=session_id)
    memory.clear()

    turn_results = []
    all_passed = True
    for turn_idx, question in enumerate(scenario["turns"]):
        streak = memory.clarification_streak()
        history_summary, recent_turns = memory.get_context()
        memory.append_turn("user", question)
        result = answer(
            question,
            history_summary=history_summary,
            recent_turns=recent_turns,
            clarification_rounds=streak,
        )
        memory.append_turn(
            "assistant", result["final_answer"], is_clarification=result["needs_clarification"]
        )

        turn_checks = [c for c in scenario["checks"] if c["turn"] == turn_idx]
        turn_pass = True
        details = []
        for check in turn_checks:
            passed, msg = check_turn(result, check)
            details.append(msg)
            if not passed:
                turn_pass = False
                all_passed = False
        turn_results.append(
            {
                "turn": turn_idx,
                "question": question,
                "final_answer": result["final_answer"],
                "needs_clarification": result["needs_clarification"],
                "passed": turn_pass,
                "details": details,
            }
        )

    memory.clear()
    return {
        "id": scenario["id"],
        "category": scenario["category"],
        "description": scenario["description"],
        "passed": all_passed,
        "turns": turn_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    with RESULTS_PATH.open("w", encoding="utf-8") as log_file:

        def out(text: str = "") -> None:
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
            log_file.write(text + "\n")
            log_file.flush()

        tally: dict[str, list[int]] = {}
        for run_idx in range(args.repeat):
            out(f"\n{'=' * 20} run {run_idx + 1}/{args.repeat} {'=' * 20}")
            for scenario in scenarios:
                result = run_scenario(scenario)
                counts = tally.setdefault(scenario["id"], [0, 0])
                counts[1] += 1
                if result["passed"]:
                    counts[0] += 1
                status = "PASS" if result["passed"] else "FAIL"
                out(f"[{status}] {scenario['id']} ({scenario['category']}): {scenario['description']}")
                for t in result["turns"]:
                    t_status = "ok" if t["passed"] else "FAIL"
                    out(f"    turn {t['turn']} [{t_status}]: Q={t['question']!r}")
                    if not t["passed"]:
                        for d in t["details"]:
                            out(f"      - {d}")
                        out(f"      answer: {t['final_answer'][:200]!r}")

        out(f"\n{'=' * 20} summary ({args.repeat} run(s)) {'=' * 20}")
        total_pass = 0
        total = 0
        for sid, (p, t) in tally.items():
            total_pass += p
            total += t
            out(f"  {sid}: {p}/{t} passed")
        out(f"\nOverall: {total_pass}/{total} scenario-runs passed ({total_pass / total:.0%})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("agent").setLevel(logging.WARNING)  # keep eval output clean; INFO for full trace
    main()
