"""One-click Phase 0 data pipeline: Atlas fetch -> servants.db -> wiki scrape -> corpus.

Usage:
    python scripts/update_all.py                 # full run, resumable (skips cached files)
    python scripts/update_all.py --force-fetch    # re-download Atlas exports even if cached
    python scripts/update_all.py --force-wiki     # re-scrape wiki pages even if cached
    python scripts/update_all.py --wiki-limit 20   # only scrape first 20 servants (smoke test)

Each step is idempotent: fetch_atlas.py and scrape_wiki.py skip files already on
disk unless told to force, so re-running this after an interruption resumes
rather than restarting from scratch.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    start = time.time()
    script, *rest = args
    result = subprocess.run([PYTHON, str(SCRIPTS_DIR / script), *rest], cwd=SCRIPTS_DIR.parent)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"!!! {label} failed (exit {result.returncode}) after {elapsed:.0f}s", flush=True)
        sys.exit(result.returncode)
    print(f"--- {label} done in {elapsed:.0f}s ---", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-fetch", action="store_true", help="re-download Atlas JSON exports")
    parser.add_argument("--force-wiki", action="store_true", help="re-scrape wiki pages")
    parser.add_argument("--wiki-limit", type=int, default=None, help="only scrape first N servants")
    args = parser.parse_args()

    fetch_args = ["fetch_atlas.py"]
    if args.force_fetch:
        fetch_args.append("--force")
    run_step("1/4 fetch_atlas", fetch_args)

    run_step("2/4 build_servants_db", ["build_servants_db.py"])

    scrape_args = ["scrape_wiki.py"]
    if args.force_wiki:
        scrape_args.append("--force")
    if args.wiki_limit:
        scrape_args += ["--limit", str(args.wiki_limit)]
    run_step("3/4 scrape_wiki", scrape_args)

    run_step("4/4 build_corpus", ["build_corpus.py"])

    print("\nAll steps complete. data/servants.db and data/corpus/servants.jsonl are up to date.")


if __name__ == "__main__":
    main()
