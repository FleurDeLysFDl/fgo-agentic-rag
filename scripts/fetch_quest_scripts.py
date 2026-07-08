"""Fetch FGO quest/story script text from Atlas Academy for the "full story
text" corpus (main story, Lostbelts, Ordeal Call, 幕间物语/interludes, and all
limited-time event/collab stories -- everything except pure farming stages,
see QUEST_EXCLUDE_WAR_IDS in config.py).

Key discovery this is built on: Atlas Academy's CN `nice_war` export already
embeds each quest's `phaseScripts` (script file ids + CDN urls) directly in
the war/spot/quest listing -- so building the quest manifest costs ZERO extra
API calls beyond the nice_war.json export we already have cached. The only
per-servant-specific script not reachable via the war listing is
`valentineScript` (pulled straight from nice_servant.json). A small number of
a servant's relateQuestIds may reference quests outside our included wars (or
not show up in the export for some other reason) -- those are fetched
individually as a fallback.

Usage:
    python scripts/fetch_quest_scripts.py [--skip-download] [--force-refetch-json]
    python scripts/fetch_quest_scripts.py --region JP --jp-exclusive-only
        -- builds a SEPARATE manifest (manifest_jp.jsonl) + script cache
        (scripts_txt_jp/) containing only quest_ids that exist in JP's
        nice_war export but not in the (already-built) CN manifest.jsonl --
        i.e. story content not yet released/ported to CN: new events, main
        story chapters, and existing servants' new interludes. This text is
        raw untranslated Japanese (Atlas Academy serves the game's own script
        files as-is, no translation), kept separate from the rest of the
        Chinese-language corpus and tagged "language": "ja" in the manifest.

Writes (region=CN, default):
  - data/quest_raw/manifest.jsonl   -- one line per quest/valentine "episode":
        {quest_id, name, war_id, war_long_name, spot_name, linked_servants,
         phases: [{phase, scripts: [{script_id, url}]}]}
  - data/quest_raw/scripts_txt/{script_id}.txt  -- raw script file cache,
        skipped if already present (resumable).

Writes (--region JP --jp-exclusive-only):
  - data/quest_raw/manifest_jp.jsonl (same shape, plus "language": "ja")
  - data/quest_raw/scripts_txt_jp/{script_id}.txt
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ATLAS_API_BASE,
    QUEST_EXCLUDE_WAR_IDS,
    QUEST_RAW_DIR,
    QUEST_SCRIPTS_DIR,
    QUEST_SCRIPTS_DIR_JP,
    RAW_DIR,
)

MANIFEST_PATH = QUEST_RAW_DIR / "manifest.jsonl"
MANIFEST_PATH_JP = QUEST_RAW_DIR / "manifest_jp.jsonl"

USER_AGENT = "fgo-agentic-rag-research-bot/0.1 (personal portfolio project)"
DOWNLOAD_WORKERS = 8


def _get_with_retry(session: requests.Session, url: str, max_retries: int = 3, **kwargs):
    kwargs.setdefault("timeout", 20)
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = 2**attempt
            print(f"  [retry] GET {url} failed ({exc.__class__.__name__}), retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise last_exc


def ensure_exports(session: requests.Session, region: str, force: bool) -> tuple[list, list]:
    war_path = RAW_DIR / f"{region.lower()}_nice_war.json"
    servant_path = RAW_DIR / f"{region.lower()}_nice_servant.json"
    if force or not war_path.exists():
        print(f"downloading nice_war export ({region})...", flush=True)
        resp = _get_with_retry(session, f"{ATLAS_API_BASE}/export/{region}/nice_war.json", timeout=120)
        war_path.write_bytes(resp.content)
    if force or not servant_path.exists():
        print(f"downloading nice_servant export ({region})...", flush=True)
        resp = _get_with_retry(session, f"{ATLAS_API_BASE}/export/{region}/nice_servant.json", timeout=120)
        servant_path.write_bytes(resp.content)

    wars = json.loads(war_path.read_text(encoding="utf-8"))
    servants = json.loads(servant_path.read_text(encoding="utf-8"))
    return wars, servants


def build_quest_manifest(wars: list) -> dict:
    """quest_id (int) -> manifest entry, for every quest in an included war
    that actually has script content."""
    manifest = {}
    for w in wars:
        if w["id"] in QUEST_EXCLUDE_WAR_IDS:
            continue
        war_long_name = (w.get("longName") or "").replace("\n", "/")
        for sp in w.get("spots", []):
            spot_name = sp.get("name")
            for q in sp.get("quests", []):
                phase_scripts = q.get("phaseScripts") or []
                if not phase_scripts:
                    continue
                phases = [
                    {
                        "phase": ps.get("phase"),
                        "scripts": [
                            {"script_id": s["scriptId"], "url": s["script"]}
                            for s in ps.get("scripts", [])
                        ],
                    }
                    for ps in phase_scripts
                ]
                manifest[q["id"]] = {
                    "quest_id": str(q["id"]),
                    "name": q.get("name") or "",
                    "war_id": w["id"],
                    "war_long_name": war_long_name,
                    "spot_name": spot_name,
                    "linked_servants": [],
                    "phases": phases,
                }
    return manifest


def add_valentine_entries(manifest: dict, servants: list) -> None:
    for s in servants:
        # A handful of servants have multiple valentineScript entries (e.g.
        # re-releases/costume-specific variants across different years) --
        # each is a genuinely distinct script, so index the key rather than
        # keying on svt_id alone or later entries silently overwrite earlier
        # ones.
        for i, v in enumerate(s.get("valentineScript") or []):
            qid = f"valentine_{s['id']}_{i}"
            manifest[qid] = {
                "quest_id": qid,
                "name": f"{s.get('name')}／情人节剧情",
                "war_id": None,
                "war_long_name": "情人节剧情",
                "spot_name": None,
                "linked_servants": [s.get("name")],
                "phases": [{"phase": 1, "scripts": [{"script_id": v["scriptId"], "url": v["script"]}]}],
            }


def link_servant_quests(session: requests.Session, manifest: dict, servants: list, region: str = "CN") -> None:
    """Tag quests already in the manifest with which servant(s) they belong to
    (interludes/ascension quests), and fetch any relateQuestId that's missing
    from the war-scan manifest entirely as a fallback."""
    missing_ids = set()
    for s in servants:
        name = s.get("name")
        for qid in s.get("relateQuestIds") or []:
            if qid in manifest:
                if name not in manifest[qid]["linked_servants"]:
                    manifest[qid]["linked_servants"].append(name)
            else:
                missing_ids.add(qid)

    if not missing_ids:
        return

    print(f"fetching {len(missing_ids)} relateQuestIds not found in war export (fallback)...", flush=True)
    fetched = 0
    for qid in tqdm(sorted(missing_ids), desc="fallback relateQuestIds", unit="quest"):
        try:
            resp = _get_with_retry(session, f"{ATLAS_API_BASE}/nice/{region}/quest/{qid}")
        except requests.exceptions.RequestException as exc:
            tqdm.write(f"  [skip] quest {qid}: {exc.__class__.__name__}")
            continue
        d = resp.json()
        phase_scripts = d.get("phaseScripts") or []
        if not phase_scripts:
            continue
        phases = [
            {
                "phase": ps.get("phase"),
                "scripts": [
                    {"script_id": sc["scriptId"], "url": sc["script"]}
                    for sc in ps.get("scripts", [])
                ],
            }
            for ps in phase_scripts
        ]
        manifest[qid] = {
            "quest_id": str(qid),
            "name": d.get("name") or "",
            "war_id": d.get("warId"),
            "war_long_name": (d.get("warLongName") or "").replace("\n", "/"),
            "spot_name": d.get("spotName"),
            "linked_servants": [],
            "phases": phases,
        }
        fetched += 1
        time.sleep(0.1)

    # second pass: link servants to the newly-fetched fallback quests
    for s in servants:
        name = s.get("name")
        for qid in s.get("relateQuestIds") or []:
            if qid in manifest and name not in manifest[qid]["linked_servants"]:
                if qid in missing_ids:
                    manifest[qid]["linked_servants"].append(name)
    print(f"  fetched {fetched}/{len(missing_ids)} fallback quests with script content", flush=True)


def collect_script_targets(manifest: dict) -> dict:
    """script_id -> url, deduped across every quest/phase."""
    targets = {}
    for entry in manifest.values():
        for phase in entry["phases"]:
            for sc in phase["scripts"]:
                targets[sc["script_id"]] = sc["url"]
    return targets


def download_one(session: requests.Session, script_id: str, url: str, scripts_dir: Path) -> tuple[str, bool, str]:
    out_path = scripts_dir / f"{script_id}.txt"
    if out_path.exists():
        return (script_id, True, "cached")
    try:
        resp = _get_with_retry(session, url, max_retries=3)
    except requests.exceptions.RequestException as exc:
        return (script_id, False, exc.__class__.__name__)
    out_path.write_bytes(resp.content)
    return (script_id, True, "downloaded")


def download_all(targets: dict, scripts_dir: Path) -> None:
    total = len(targets)
    ok = 0
    cached = 0
    failed = []
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_one, session, sid, url, scripts_dir): sid for sid, url in targets.items()}
        with tqdm(total=total, desc="downloading scripts", unit="file") as bar:
            for fut in as_completed(futures):
                script_id, success, status = fut.result()
                if success:
                    ok += 1
                    if status == "cached":
                        cached += 1
                else:
                    failed.append((script_id, status))
                bar.set_postfix(ok=ok, cached=cached, failed=len(failed))
                bar.update(1)

    print(f"done: {ok}/{total} ok ({cached} already cached), {len(failed)} failed", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} scripts failed to download:", flush=True)
        for sid, status in failed[:20]:
            print(f"  {sid}: {status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="only (re)build manifest.jsonl, skip script downloads")
    parser.add_argument("--force-refetch-json", action="store_true", help="re-download nice_war/nice_servant exports even if cached")
    parser.add_argument("--region", default="CN", help="Atlas Academy region to pull the war/servant export from (default CN)")
    parser.add_argument(
        "--jp-exclusive-only",
        action="store_true",
        help="keep only quest_ids from --region JP that are NOT already in the existing "
        "CN manifest.jsonl (new events/main-story/interludes not yet ported to CN) -- "
        "writes to manifest_jp.jsonl / scripts_txt_jp/ instead, tagged language='ja'",
    )
    args = parser.parse_args()

    if args.jp_exclusive_only and args.region == "CN":
        raise SystemExit("--jp-exclusive-only only makes sense together with --region JP")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    wars, servants = ensure_exports(session, args.region, args.force_refetch_json)
    print(f"loaded {len(wars)} wars, {len(servants)} servants ({args.region})", flush=True)

    manifest = build_quest_manifest(wars)
    print(f"quest manifest (war-scan, script content only): {len(manifest)} quests", flush=True)

    add_valentine_entries(manifest, servants)
    print(f"quest manifest after adding valentine scripts: {len(manifest)} entries", flush=True)

    link_servant_quests(session, manifest, servants, region=args.region)

    manifest_path = MANIFEST_PATH
    scripts_dir = QUEST_SCRIPTS_DIR
    if args.jp_exclusive_only:
        if not MANIFEST_PATH.exists():
            raise SystemExit(f"{MANIFEST_PATH} not found -- run the default (CN) pass first")
        cn_quest_ids = set()
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            for line in f:
                cn_quest_ids.add(json.loads(line)["quest_id"])
        manifest = {qid: entry for qid, entry in manifest.items() if str(qid) not in cn_quest_ids}
        for entry in manifest.values():
            entry["language"] = "ja"
        print(f"filtered to {len(manifest)} JP-exclusive entries (not present in CN manifest)", flush=True)
        manifest_path = MANIFEST_PATH_JP
        scripts_dir = QUEST_SCRIPTS_DIR_JP

    QUEST_RAW_DIR.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path} ({len(manifest)} entries)", flush=True)

    targets = collect_script_targets(manifest)
    print(f"unique script files referenced: {len(targets)}", flush=True)

    if args.skip_download:
        print("--skip-download set, not downloading script files.", flush=True)
        return

    download_all(targets, scripts_dir)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
