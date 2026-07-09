"""Servant nickname/alias resolution: maps informal names players actually
type (fan nicknames, alternate transliterations, English community slang,
e.g. "红A", "Jalter", "呆毛王") to the corpus's own canonical title string
(matching corpus_text/retrieval source naming, e.g. "卫宫", "贞德〔Alter〕",
"阿尔托莉雅·潘德拉贡(Lancer)"). Applied to a question's text before
decomposition (agent/graph.py's decompose) so downstream sub-question
extraction, structured lookup, and retrieval all see a name the corpus
actually recognizes instead of a nickname that matches nothing.

data/servant_aliases.json is a flat {alias: canonical_name} dict, seeded
2026-07 from public FGO community nickname round-ups (Chinese: Mooncell/
gamersky/xiaomi compilations; English: gcores' write-up of the global
community's own nicknames) and cross-checked against every canonical name
against the actual corpus titles. Meant to be hand-maintained/extended
directly -- no build step needed, just add/edit entries.
"""

import json
from pathlib import Path

ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "servant_aliases.json"


def _load_aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    data = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    data.pop("_comment", None)
    return data


_ALIASES = _load_aliases()
# Longest alias first, so a longer alias is substituted whole before a
# shorter alias that happens to be one of its substrings gets a chance to
# split it apart (e.g. "黑无毛" before "无毛" would-be-collision avoidance).
_SORTED_ALIASES = sorted(_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True)


def resolve_aliases(text: str) -> str:
    """Replace every known alias occurrence in text with its canonical name.
    Plain substring replacement -- aliases here are curated to be distinctive
    enough that false-positive matches inside unrelated words aren't expected
    to be a practical problem for this FGO-only corpus."""
    for alias, canonical in _SORTED_ALIASES:
        if alias in text:
            text = text.replace(alias, canonical)
    return text
