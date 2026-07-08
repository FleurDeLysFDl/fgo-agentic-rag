"""Parse Atlas Academy's raw FGO quest script format (the .txt files served from
static.atlasacademy.io/{region}/Script/...) into structured dialogue turns.

The raw format looks like:

    ＄91-10-01-01-1-0

    [soundStopAll]
    [bgm BGM_EVENT_5 0.1]
    [scene 10710]
    [charaSet A 8001000 2 玛修]
    [charaFace D 4]
    ＠阿尔托莉雅
    太惨烈了。[r]不经历个十年二十年，这火焰是不会熄灭的吧。
    [k]

    ？1：一定是由于圣杯的缘故

    ？2：因为这里成了特异点

    ？！

Rules applied here:
  - ＄... header line -> dropped (internal script id, not content).
  - Any line that, once stripped, is entirely one or more `[bracket commands]`
    (bgm/scene/chara sprite/wait/etc engine directives) -> dropped whole.
  - ＠Name line -> sets the "current speaker" for the dialogue text that follows,
    until the next ＠ line, a blank line, or end of script.
  - Plain text lines -> dialogue for the current speaker. Inline `[r]` (manual
    line break within one textbox) is converted to a newline; three families
    of "variable substitution" bracket tags carry real, otherwise-unrecoverable
    dialogue content and are resolved to a single displayed value rather than
    stripped (see `_resolve_variable_tag`/regexes below); any other inline
    bracket tag is stripped out entirely (pure engine directives -- sprite/
    sound/camera/branch commands -- confirmed by surveying the ~700 distinct
    tag names appearing across the whole downloaded corpus).
  - `？N：option text` lines -> a player dialogue-choice option (speaker=None,
    kept as its own turn so branches aren't lost).
  - `？！` (choice-block terminator) -> dropped, carries no text.
  - Narration text that appears with no preceding ＠ line -> kept as a turn
    with speaker=None (e.g. onscreen narration captions).

Variable-substitution bracket tags (found via corpus-wide survey, all were
being silently deleted by the old generic-strip-only approach -- a real
content-loss bug, since these are colon-separated "fields" with the naive
generic-bracket-content match, not pure directives):
  - `[%N]`               -- Master-name variable; no way to recover the
                             player's actual chosen name, so substituted with
                             a generic term for one's Master in FGO -- "御主"
                             for zh scripts, "マスター" for ja scripts (see
                             `language` param on parse_script/render_script).
  - `[#A:B]` / `[#A]`     -- ruby/gloss annotation: `A` is the primary displayed
                             text, `B` (if present) is a secondary reading/gloss
                             (translation, pronunciation, or in some cases the
                             true name behind an alias) -- keep `A`.
  - `[&A:B]`              -- protagonist-gender text branch: `A` is one gender's
                             phrasing, `B` the other's (e.g. `[&Mr.:Lady]`,
                             `[&Sir:Ma'am]`, or `[&:先生]` where the unmarked
                             gender gets no suffix at all) -- since the actual
                             chosen gender isn't recoverable, we deterministically
                             keep the first non-empty field for consistency.
  - `[servantName ID:A:B]` -- an in-quest alias for a servant: `A` is the alias
                             actually displayed at that point in the story
                             (`B` is usually the real name, for reference) --
                             keep `A`.
"""

import re

INLINE_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
MASTER_PLACEHOLDER_RE = re.compile(r"\[%\d+\]")
RUBY_ANNOTATION_RE = re.compile(r"\[#([^\]]*)\]")
GENDER_BRANCH_RE = re.compile(r"\[&([^\]]*)\]")
SERVANT_ALIAS_RE = re.compile(r"\[servantName\s+\d+:([^\]]*)\]")
# Three speaker-line conventions observed in the wild:
#   ＠阿尔托莉雅            -- name directly
#   ＠C：？？？             -- charaSet slot letter (matches [charaSet C ...])
#                              followed by the actual display name
#   ＠                       -- bare, no name at all: switches to "no speaker"
#                              (narrator/on-screen text, e.g. a displayed
#                              letter) for the turns that follow, until the
#                              next ＠ line -- NOT literal "＠" text content.
SPEAKER_SLOT_LINE_RE = re.compile(r"^[＠@][A-Za-z]{1,2}[:：](.+)$")
SPEAKER_LINE_RE = re.compile(r"^[＠@](.*)$")
# `？N：text` is the common case, but a sizeable minority of choice lines carry
# one or more internal `,ref` fields between the option number and the colon
# (e.g. `？1,030052111：text`, `？1,1030,saveMaterial：text` -- branch-target
# script/flag ids, confirmed via corpus-wide survey). These aren't content and
# are discarded; only the number and the trailing text matter.
CHOICE_LINE_RE = re.compile(r"^？\s*\d+(?:\s*,[^:：]*)*\s*[:：]\s*(.*)$")
CHOICE_END_RE = re.compile(r"^？\s*[!！]\s*$")
# `？？,ref：不选择：<SE cue params>` -- the "player let the timer run out"
# branch for timed choices; carries no player-facing text, just a sound-effect
# cue, so it's dropped like CHOICE_END_RE rather than parsed as an option.
CHOICE_TIMEOUT_RE = re.compile(r"^？+\s*,.*不选择.*$")
HEADER_LINE_RE = re.compile(r"^[＄$]")


def _first_nonempty_field(content: str) -> str:
    """For `:`-delimited variable-substitution tag content (ruby/gender/alias
    tags), return the first non-empty field, falling back to the last field
    in the rare case where every field is empty (e.g. `[&:]`)."""
    fields = content.split(":")
    for field in fields:
        if field.strip():
            return field
    return fields[-1] if fields else ""


MASTER_PLACEHOLDER_BY_LANG = {"zh": "御主", "ja": "マスター"}
CHOICE_PREFIX_BY_LANG = {"zh": "（选项）", "ja": "（選択肢）"}


def _clean_text(line: str, language: str = "zh") -> str:
    """Strip inline bracket tags from a dialogue line, turning [r] into a
    newline (manual in-textbox line break), resolving the variable-
    substitution tag families ([%N], [#A:B], [&A:B], [servantName ID:A:B] --
    see module docstring) to a single displayed value, and dropping
    everything else (including inline color tags like [51d4ff]...[-], which
    also show up wrapping speaker names, hence _clean_text is applied to the
    speaker capture too).

    `language` picks the substitution term for [%N] ("御主" for zh, "マスター"
    for ja scripts) so JP-exclusive content doesn't get a Chinese term baked
    into otherwise-untranslated Japanese text."""
    line = line.replace("[r]", "\n")
    line = MASTER_PLACEHOLDER_RE.sub(MASTER_PLACEHOLDER_BY_LANG.get(language, "御主"), line)
    line = SERVANT_ALIAS_RE.sub(lambda m: _first_nonempty_field(m.group(1)), line)
    line = RUBY_ANNOTATION_RE.sub(lambda m: _first_nonempty_field(m.group(1)), line)
    line = GENDER_BRANCH_RE.sub(lambda m: _first_nonempty_field(m.group(1)), line)
    line = INLINE_BRACKET_RE.sub("", line)
    return line.strip()


def parse_script(raw_text: str, language: str = "zh") -> list[dict]:
    """Parse one raw script .txt's contents into a list of
    {"speaker": str | None, "text": str} dialogue turns, in original order."""
    turns: list[dict] = []
    current_speaker = None
    buffer: list[str] = []
    choice_prefix = CHOICE_PREFIX_BY_LANG.get(language, "（选项）")

    def flush():
        nonlocal buffer
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                turns.append({"speaker": current_speaker, "text": text})
            buffer = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if HEADER_LINE_RE.match(line):
            continue

        choice_end = CHOICE_END_RE.match(line)
        if choice_end:
            continue

        if CHOICE_TIMEOUT_RE.match(line):
            continue

        choice = CHOICE_LINE_RE.match(line)
        if choice:
            flush()
            option_text = _clean_text(choice.group(1), language)
            if option_text:
                turns.append({"speaker": None, "text": f"{choice_prefix}{option_text}"})
            continue

        speaker = SPEAKER_SLOT_LINE_RE.match(line) or SPEAKER_LINE_RE.match(line)
        if speaker:
            flush()
            # a bare ＠ (no name, or a name that cleans to nothing) means
            # "no current speaker" -- narrator/on-screen text follows.
            current_speaker = _clean_text(speaker.group(1), language) or None
            continue

        cleaned = _clean_text(line, language)
        if cleaned:
            buffer.append(cleaned)

    flush()
    return turns


def format_turn(turn: dict) -> str:
    speaker = turn.get("speaker")
    text = turn.get("text") or ""
    if speaker:
        return f"{speaker}：{text}"
    return text


def render_script(raw_text: str, language: str = "zh") -> str:
    """Convenience: parse a raw script and render it back to a single
    human-readable text block (one line per dialogue turn)."""
    return "\n".join(format_turn(t) for t in parse_script(raw_text, language))
