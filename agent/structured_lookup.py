"""Read-only structured lookup over data/servants.db.

Exposes a single parameterized query function (no free-form SQL execution)
so the agent can answer stat-style questions (skills, NP card type/rank,
rarity, acquisition method) without hallucinating from lore text.

Usage (library):
    from agent.structured_lookup import lookup_servant
    lookup_servant("阿尔托莉雅·潘德拉贡", class_hint="lancer")
"""

import sqlite3
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SERVANTS_DB_PATH


def _row_to_dict(cur: sqlite3.Cursor, row: tuple) -> dict:
    return {desc[0]: value for desc, value in zip(cur.description, row)}


def lookup_servant(name: str, class_hint: str | None = None, limit: int = 10) -> list[dict]:
    """Fuzzy-match a servant by Chinese name (optionally narrowed by class),
    returning each match's basic info, skills, and noble phantasms.

    Returns a list because name_cn is not unique across class-swap /
    costume variants (e.g. multiple 阿尔托莉雅·潘德拉贡 rows for Saber/
    Lancer/Rider/Ruler forms) -- callers should disambiguate using the
    returned class_name, or present all matches if still ambiguous.
    """
    conn = sqlite3.connect(str(SERVANTS_DB_PATH))
    try:
        cur = conn.cursor()

        # Exact name_cn match first: substring matching alone can't tell
        # "阿尔托莉雅·潘德拉贡" (base) apart from "阿尔托莉雅·潘德拉贡〔Alter〕"
        # (which also contains the base name as a substring) when both
        # happen to share the same class_hint -- only fall back to fuzzy
        # substring matching if the exact name isn't in the database.
        exact_query = "SELECT * FROM servants WHERE name_cn = ?"
        exact_params: list = [name]
        if class_hint:
            exact_query += " AND class_name LIKE ?"
            exact_params.append(f"%{class_hint}%")
        exact_query += " LIMIT ?"
        exact_params.append(limit)
        cur.execute(exact_query, exact_params)
        servant_rows = [_row_to_dict(cur, row) for row in cur.fetchall()]

        if not servant_rows:
            query = "SELECT * FROM servants WHERE name_cn LIKE ?"
            params: list = [f"%{name}%"]
            if class_hint:
                query += " AND class_name LIKE ?"
                params.append(f"%{class_hint}%")
            query += " LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            servant_rows = [_row_to_dict(cur, row) for row in cur.fetchall()]

        results = []
        for servant in servant_rows:
            svt_id = servant["svt_id"]

            cur.execute(
                "SELECT skill_num, name_en, detail_en FROM skills WHERE svt_id = ? ORDER BY skill_num",
                (svt_id,),
            )
            skills = [_row_to_dict(cur, row) for row in cur.fetchall()]

            cur.execute(
                "SELECT name_en, card_type, rank, detail_en FROM noble_phantasms WHERE svt_id = ?",
                (svt_id,),
            )
            nps = [_row_to_dict(cur, row) for row in cur.fetchall()]

            results.append({**servant, "skills": skills, "noble_phantasms": nps})
        return results
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--class-hint", default=None)
    args = parser.parse_args()

    print(json.dumps(lookup_servant(args.name, args.class_hint), ensure_ascii=False, indent=2))
