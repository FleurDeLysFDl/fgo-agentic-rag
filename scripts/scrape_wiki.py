"""Scrape servant profile text ("资料") from the Mooncell FGO wiki (fgo.wiki).

Uses the MediaWiki API (action=query, prop=revisions) to pull raw wikitext --
much lighter on the site than rendering full HTML pages, and lets us parse
the structured `{{个人资料|...}}` template directly instead of scraping markup.

Matching: Atlas servant CN names are usually unique wiki page titles, except
for class-swap variants that share an identical base name (e.g. four
different-class "阿尔托莉雅·潘德拉贡" servants). For those we verify the
page's own `{{基础数值|...|职阶=...}}` field against our expected class, and
retry with a "(ClassName)" suffix (Mooncell's disambiguation convention) if it
doesn't match.

Servants not yet released on CN (`servants.db.name_cn IS NULL`) have no
official Chinese name to search by, but Mooncell still documents JP-exclusive
content ahead of CN release under a placeholder title "从者{collection_no}"
(e.g. "从者456") until the servant's true identity/name is revealed -- see
`_unreleased_title_candidates()`.

Usage:
    python scripts/scrape_wiki.py [--limit N] [--force]

Writes one JSON file per resolved servant to data/wiki_raw/{svt_id}.json and
backfills servants.db's `acquisition` column from each page's 获取途径 field
(and `name_cn` too, for previously-CN-unreleased servants that just resolved).
"""

import argparse
import json
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import mwparserfromhell
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SERVANTS_DB_PATH, WIKI_API_BASE, WIKI_REQUEST_DELAY_SECONDS, WIKI_USER_AGENT, WIKI_RAW_DIR

CLASS_DISPLAY_MAP = {
    "alterEgo": "Alter Ego",
    "moonCancer": "Moon Cancer",
    "beastEresh": "Beast",
    "unBeastOlgaMarie": "Beast",
}


def class_display_name(class_name: str) -> str:
    return CLASS_DISPLAY_MAP.get(class_name, class_name.capitalize())


def normalize_class(s: str) -> str:
    """Mooncell writes multi-word classes inconsistently ("Alter Ego" in URLs
    but "Alterego" in the infobox 职阶 field) -- compare with spaces stripped."""
    return s.lower().replace(" ", "")


def fetch_wikitext(session: requests.Session, title: str, max_retries: int = 3) -> dict | None:
    # This is meant to run unattended for a long time (~400+ servants, several
    # requests each) -- a bare requests call has no retry, so a single
    # transient SSL-handshake/read timeout (observed in practice against
    # fgo.wiki) would crash the entire multi-hour run. Retry with backoff
    # instead of propagating the first hiccup.
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.get(
                WIKI_API_BASE,
                params={
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "titles": title,
                    "redirects": 1,
                    "format": "json",
                },
                timeout=20,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"  [retry] fetch {title!r} failed ({exc.__class__.__name__}), retrying in {wait}s...", flush=True)
            time.sleep(wait)
    else:
        raise last_exc

    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "revisions" not in page:
        return None
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "wikitext": page["revisions"][0]["slots"]["main"]["*"],
    }


def extract_templates(wikicode, name: str):
    return wikicode.filter_templates(matches=lambda t: t.name.strip() == name)


def parse_page(wikitext: str, expected_class: str) -> dict:
    wikicode = mwparserfromhell.parse(wikitext)

    # Dual-class servants (e.g. Henry Jekyll & Hyde) can carry two 基础数值
    # blocks on one page, one per class form -- pick the one matching Atlas's
    # class for this svt_id, falling back to the first if none match.
    acquisition = None
    class_field = None
    stats_tpls = extract_templates(wikicode, "基础数值")
    stats_tpl = None
    for tpl in stats_tpls:
        if tpl.has("职阶") and normalize_class(str(tpl.get("职阶").value).strip()) == normalize_class(expected_class):
            stats_tpl = tpl
            break
    if stats_tpl is None and stats_tpls:
        stats_tpl = stats_tpls[0]
    if stats_tpl is not None and stats_tpl.has("获取途径"):
        acquisition = str(stats_tpl.get("获取途径").value).strip()
    if stats_tpl is not None and stats_tpl.has("职阶"):
        class_field = str(stats_tpl.get("职阶").value).strip()

    profile_parts = []
    profile_tpls = extract_templates(wikicode, "个人资料")
    profile_tpl = profile_tpls[0] if profile_tpls else None
    if profile_tpl is not None:
        if profile_tpl.has("详情"):
            text = str(profile_tpl.get("详情").value).strip()
            if text:
                profile_parts.append(text)
        i = 1
        while profile_tpl.has(f"资料{i}") or profile_tpl.has(f"资料{i}条件"):
            if profile_tpl.has(f"资料{i}"):
                text = str(profile_tpl.get(f"资料{i}").value).strip()
                if text:
                    profile_parts.append(text)
            i += 1

    # April Fools alt-profile ("愚人节资料") -- a page can carry several of
    # these (one per joke-costume variant, e.g. "通常"/"通常(奥特瑙斯)"), each
    # a simple {中文=..., 日文=...} pair (no numbered 资料N fields like the
    # main profile template). We only want the Chinese text.
    april_fools_parts = []
    for tpl in extract_templates(wikicode, "愚人节资料"):
        if tpl.has("中文"):
            text = str(tpl.get("中文").value).strip()
            if text:
                april_fools_parts.append(text)

    return {
        "acquisition": acquisition,
        "class_field": class_field,
        "profile_parts": profile_parts,
        "april_fools_parts": april_fools_parts,
    }


def parse_voice_page(wikitext: str) -> dict:
    """Parse a Mooncell "{name}/语音" (or linked sub-)page: each
    {{#invoke:VoiceTable|table|...}} block groups a set of numbered
    标题N/日文N/中文N/条件N/语音N fields under one 表格标题 (table title,
    e.g. "战斗", "个人空间"). We keep 标题N (line label), 中文N (the actual
    Chinese line -- what we want for the corpus) and 条件N (context, e.g.
    "觉醒前"/"觉醒后" pre/post-ascension); 日文N (Japanese) and 语音N (audio
    filename) are dropped as not useful for a Chinese-language text corpus.

    Also collects {{参阅|Title|Description}} references so the caller can
    follow linked voice sub-pages (e.g. "{name}/御主任务语音",
    "{name}/情人节剧情语音") that aren't transcluded on the main voice page.
    """
    wikicode = mwparserfromhell.parse(wikitext)
    sections = []
    for tpl in wikicode.filter_templates(matches=lambda t: t.name.strip().lower().startswith("#invoke:voicetable")):
        title = None
        if tpl.has("表格标题"):
            # The raw value can trail into an embedded <noinclude>|可播放=1</noinclude>
            # tag (its inner "|" isn't a real template-argument separator, but
            # mwparserfromhell keeps it as part of this argument's raw value) --
            # only the first line is the actual title text.
            title = str(tpl.get("表格标题").value).strip().splitlines()[0].strip()

        lines = []
        i = 1
        while tpl.has(f"标题{i}") or tpl.has(f"中文{i}"):
            label = str(tpl.get(f"标题{i}").value).strip() if tpl.has(f"标题{i}") else ""
            text = str(tpl.get(f"中文{i}").value).strip() if tpl.has(f"中文{i}") else ""
            condition = str(tpl.get(f"条件{i}").value).strip() if tpl.has(f"条件{i}") else ""
            if text:
                lines.append({"label": label, "text": text, "condition": condition})
            i += 1
        if lines:
            sections.append({"title": title, "lines": lines})

    see_also = []
    for tpl in wikicode.filter_templates(matches=lambda t: t.name.strip() in ("参阅", "参阅2", "参阅三")):
        if not tpl.params:
            continue
        target = str(tpl.params[0].value).strip()
        # Only follow plain ".../语音" sub-page links (skip things like
        # "{{PAGENAME}}/从者任务" interlude-quest links or template-valued
        # params such as {{BiliSearch|...}} -- those aren't voice pages).
        if target and not target.startswith("{{") and target.endswith("语音"):
            see_also.append(target)

    return {"sections": sections, "see_also": see_also}


def fetch_all_voice_lines(session: requests.Session, base_title: str) -> list[dict]:
    """Fetch "{base_title}/语音" plus any linked ".../语音" sub-pages
    referenced from within it (e.g. Master Mission voice lines, event-story
    voice lines) -- recursion is bounded by a seen-set, though in observed
    samples linked sub-pages don't themselves link further voice pages."""
    all_sections: list[dict] = []
    seen: set[str] = set()
    queue = [f"{base_title}/语音"]
    while queue:
        title = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        page = fetch_wikitext(session, title)
        time.sleep(WIKI_REQUEST_DELAY_SECONDS)
        if page is None:
            continue
        parsed = parse_voice_page(page["wikitext"])
        for section in parsed["sections"]:
            section["source_page"] = title
            all_sections.append(section)
        for linked in parsed["see_also"]:
            if linked not in seen:
                queue.append(linked)
    return all_sections


def _title_candidates(name_cn: str, expected_class: str) -> list[str]:
    # Atlas's CN names occasionally use full-width Latin characters (e.g.
    # "ＵＤＫ－巴格斯特") where the actual Mooncell page title uses the plain
    # half-width form ("UDK-巴格斯特") -- NFKC normalization maps these back
    # to ASCII. Keep original-form candidates first (cheap common case), add
    # the normalized form only if it differs.
    candidates = [name_cn, f"{name_cn}({expected_class})"]
    normalized = unicodedata.normalize("NFKC", name_cn)
    if normalized != name_cn:
        candidates += [normalized, f"{normalized}({expected_class})"]
    return candidates


def _unreleased_title_candidates(collection_no: int | None) -> list[str]:
    # Servants not yet released on CN have no official name_cn (Atlas's CN
    # nice_servant.json simply omits them), so there's no name to search by.
    # But Mooncell tracks JP-exclusive content ahead of CN release under a
    # placeholder title "从者{collection_no}" (e.g. "从者456") until the
    # servant's true CN name is revealed/localized, then keeps a redirect --
    # verified this resolves cleanly for 27/29 CN-unreleased servants as of
    # 2026-07-07 (collection_no 444-471). The 2 remaining have collection_no=0
    # (not yet assigned a real slot, likely unconfirmed/enemy-only datamine
    # entries) and are left unresolved.
    if not collection_no:
        return []
    return [f"从者{collection_no}"]


def resolve_and_scrape(
    session: requests.Session,
    svt_id: int,
    name_cn: str | None,
    class_name: str,
    collection_no: int | None = None,
) -> dict | None:
    expected_class = class_display_name(class_name)
    candidates = _title_candidates(name_cn, expected_class) if name_cn else _unreleased_title_candidates(collection_no)

    fallback = None  # best-effort match if no candidate's class field agrees
    for title in candidates:
        page = fetch_wikitext(session, title)
        time.sleep(WIKI_REQUEST_DELAY_SECONDS)
        if page is None:
            continue
        parsed = parse_page(page["wikitext"], expected_class)
        if parsed["class_field"] and normalize_class(parsed["class_field"]) != normalize_class(expected_class):
            # Genuine class mismatch (e.g. a servant re-released under its
            # true class after being introduced under a spoiler-placeholder
            # class like "Pretender" -- Mooncell keeps only one page, already
            # showing the revealed class, so the placeholder-class DB row can
            # never find a class-matching page). Remember the first such page
            # as a last-resort fallback instead of discarding it outright.
            if fallback is None:
                fallback = (page, parsed)
            continue

        voice_lines = fetch_all_voice_lines(session, page["title"])
        return {
            "svt_id": svt_id,
            # No official CN localization yet -- back off to the Mooncell page
            # title itself (already disambiguated, e.g. "Passionlip(Saber)")
            # so structured_lookup.py has a usable Chinese name to match on.
            "name_cn": name_cn if name_cn is not None else page["title"],
            "unreleased_in_cn": name_cn is None,
            "resolved_title": page["title"],
            "pageid": page["pageid"],
            "url": f"https://fgo.wiki/w/{page['title']}",
            "acquisition": parsed["acquisition"],
            "profile_parts": parsed["profile_parts"],
            "april_fools_parts": parsed["april_fools_parts"],
            "voice_lines": voice_lines,
        }

    if fallback is not None:
        page, parsed = fallback
        voice_lines = fetch_all_voice_lines(session, page["title"])
        return {
            "svt_id": svt_id,
            "name_cn": name_cn if name_cn is not None else page["title"],
            "unreleased_in_cn": name_cn is None,
            "resolved_title": page["title"],
            "pageid": page["pageid"],
            "url": f"https://fgo.wiki/w/{page['title']}",
            "acquisition": parsed["acquisition"],
            "profile_parts": parsed["profile_parts"],
            "april_fools_parts": parsed["april_fools_parts"],
            "voice_lines": voice_lines,
            "class_mismatch_fallback": True,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N servants (for testing)")
    parser.add_argument("--force", action="store_true", help="re-scrape even if cached")
    parser.add_argument(
        "--svt-ids",
        type=str,
        default=None,
        help="comma-separated svt_ids to (re-)scrape only these, ignoring --limit "
        "(implies --force for exactly these ids, since the whole point is to "
        "refresh already-cached-but-wrong entries)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(SERVANTS_DB_PATH)
    cur = conn.cursor()
    # Includes CN-unreleased rows (name_cn IS NULL) -- resolve_and_scrape()
    # falls back to a "从者{collection_no}" Mooncell placeholder title for
    # those instead of a name-based lookup.
    cur.execute("SELECT svt_id, name_cn, class_name, collection_no FROM servants ORDER BY collection_no")
    rows = cur.fetchall()
    if args.svt_ids:
        wanted = {int(s) for s in args.svt_ids.split(",") if s.strip()}
        rows = [r for r in rows if r[0] in wanted]
        args.force = True
    elif args.limit:
        rows = rows[: args.limit]

    session = requests.Session()
    session.headers["User-Agent"] = WIKI_USER_AGENT

    total = len(rows)
    n_ok = n_skip = n_unresolved = 0
    unresolved = []
    for idx, (svt_id, name_cn, class_name, collection_no) in enumerate(rows, 1):
        out_path = WIKI_RAW_DIR / f"{svt_id}.json"
        if out_path.exists() and not args.force:
            # build_servants_db.py rebuilds servants.db from scratch each run,
            # dropping the acquisition/name_cn backfills -- reapply both from
            # the cached scrape result instead of only writing them on a
            # fresh HTTP fetch.
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("acquisition"):
                cur.execute("UPDATE servants SET acquisition = ? WHERE svt_id = ?", (cached["acquisition"], svt_id))
            if name_cn is None and cached.get("name_cn"):
                cur.execute("UPDATE servants SET name_cn = ? WHERE svt_id = ?", (cached["name_cn"], svt_id))
            conn.commit()
            n_skip += 1
            print(f"[{idx}/{total}] [skip-cached] svt_id={svt_id} name_cn={name_cn}", flush=True)
            continue

        result = resolve_and_scrape(session, svt_id, name_cn, class_name, collection_no)
        if result is None:
            n_unresolved += 1
            unresolved.append((svt_id, name_cn, class_name))
            print(f"[{idx}/{total}] [unresolved] svt_id={svt_id} name_cn={name_cn} class={class_name}", flush=True)
            continue

        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["acquisition"]:
            cur.execute("UPDATE servants SET acquisition = ? WHERE svt_id = ?", (result["acquisition"], svt_id))
        if name_cn is None:
            cur.execute("UPDATE servants SET name_cn = ? WHERE svt_id = ?", (result["name_cn"], svt_id))
        conn.commit()
        n_ok += 1
        n_voice_lines = sum(len(sec["lines"]) for sec in result["voice_lines"])
        print(
            f"[{idx}/{total}] [ok] svt_id={svt_id} -> {result['resolved_title']} "
            f"(profile={len(result['profile_parts'])} april_fools={len(result['april_fools_parts'])} "
            f"voice_sections={len(result['voice_lines'])} voice_lines={n_voice_lines})",
            flush=True,
        )

    conn.close()
    print(f"\ndone: {n_ok} scraped, {n_skip} cached-skipped, {n_unresolved} unresolved", flush=True)
    if unresolved:
        print("unresolved servants (name-collision or missing page):")
        for svt_id, name_cn, class_name in unresolved:
            print(f"  {svt_id}\t{name_cn}\t{class_name}")


if __name__ == "__main__":
    main()
