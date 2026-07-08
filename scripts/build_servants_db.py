"""Parse cached Atlas Academy JSON exports into data/servants.db (SQLite).

Usage:
    python scripts/build_servants_db.py

Reads:
    data/raw/{region}_nice_servant_lang_en.json  (structured region, e.g. JP)
    data/raw/{region}_nice_servant.json          (name region, e.g. CN, for wiki matching)

Writes:
    data/servants.db with tables: servants, skills, noble_phantasms
    (acquisition/获取途径 is left NULL here; scripts/scrape_wiki.py fills it in
    from the Mooncell infobox template since that is the reliable source for
    permanent/limited/story-locked classification.)
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ATLAS_NAME_REGION, ATLAS_STRUCTURED_REGION, RAW_DIR, SERVANTS_DB_PATH

PLAYABLE_TYPES = {"normal", "heroine"}

# Atlas nice_servant export encodes card type as a numeric-string code rather
# than a name; verified against Mooncell's known card order for Artoria
# (cards=['3','1','1','2','2'] == Quick,Arts,Arts,Buster,Buster).
CARD_TYPE_MAP = {"1": "arts", "2": "buster", "3": "quick", "4": "extra"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS servants (
    svt_id INTEGER PRIMARY KEY,
    collection_no INTEGER,
    name_en TEXT,
    name_cn TEXT,
    class_name TEXT,
    rarity INTEGER,
    gender TEXT,
    attribute TEXT,
    traits TEXT,        -- JSON array of trait name strings
    acquisition TEXT     -- filled later from wiki: 常驻 / 限定 / 活动 / 圣杯前线 / etc
);

CREATE TABLE IF NOT EXISTS skills (
    svt_id INTEGER,
    skill_num INTEGER,
    name_en TEXT,
    detail_en TEXT,
    FOREIGN KEY (svt_id) REFERENCES servants(svt_id)
);

CREATE TABLE IF NOT EXISTS noble_phantasms (
    svt_id INTEGER,
    name_en TEXT,
    card_type TEXT,          -- buster / arts / quick
    rank TEXT,
    individuality_traits TEXT,  -- JSON array; presence of 'attribute*' entries = NP has
                                 -- an elemental attribute lock, absence = "no attribute" NP
    detail_en TEXT,
    FOREIGN KEY (svt_id) REFERENCES servants(svt_id)
);
"""


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    jp_path = RAW_DIR / f"{ATLAS_STRUCTURED_REGION.lower()}_nice_servant_lang_en.json"
    cn_path = RAW_DIR / f"{ATLAS_NAME_REGION.lower()}_nice_servant.json"
    if not jp_path.exists() or not cn_path.exists():
        raise SystemExit("Missing cached Atlas exports. Run scripts/fetch_atlas.py first.")

    structured = load_json(jp_path)
    names = load_json(cn_path)
    name_cn_by_id = {d["id"]: d["name"] for d in names}

    SERVANTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SERVANTS_DB_PATH.exists():
        SERVANTS_DB_PATH.unlink()
    conn = sqlite3.connect(SERVANTS_DB_PATH)
    conn.executescript(SCHEMA)

    n_servants = n_skills = n_nps = 0
    for d in structured:
        if d["type"] not in PLAYABLE_TYPES:
            continue

        svt_id = d["id"]
        trait_names = [t["name"] for t in d.get("traits", [])]

        conn.execute(
            """INSERT INTO servants
               (svt_id, collection_no, name_en, name_cn, class_name, rarity,
                gender, attribute, traits, acquisition)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                svt_id,
                d.get("collectionNo"),
                d.get("name"),
                name_cn_by_id.get(svt_id),
                d.get("className"),
                d.get("rarity"),
                d.get("gender"),
                d.get("attribute"),
                json.dumps(trait_names, ensure_ascii=False),
            ),
        )
        n_servants += 1

        for skill in d.get("skills", []):
            if skill.get("num") is None:
                continue
            conn.execute(
                "INSERT INTO skills (svt_id, skill_num, name_en, detail_en) VALUES (?, ?, ?, ?)",
                (svt_id, skill["num"], skill.get("name"), skill.get("detail")),
            )
            n_skills += 1

        # noblePhantasms often has multiple strength-upgrade entries per NP slot;
        # keep the highest-priority (current/live) version per distinct NP name.
        seen_np_names = set()
        for np in sorted(d.get("noblePhantasms", []), key=lambda x: x.get("priority", 0), reverse=True):
            name = np.get("name")
            if name in seen_np_names:
                continue
            seen_np_names.add(name)
            trait_names = [t["name"] for t in np.get("individuality", [])]
            conn.execute(
                """INSERT INTO noble_phantasms
                   (svt_id, name_en, card_type, rank, individuality_traits, detail_en)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    svt_id,
                    name,
                    CARD_TYPE_MAP.get(np.get("card"), np.get("card")),
                    np.get("rank"),
                    json.dumps(trait_names, ensure_ascii=False),
                    np.get("detail"),
                ),
            )
            n_nps += 1

    conn.commit()
    conn.close()
    print(f"wrote {SERVANTS_DB_PATH}: {n_servants} servants, {n_skills} skills, {n_nps} noble phantasms")


if __name__ == "__main__":
    main()
