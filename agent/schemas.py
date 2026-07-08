"""Pydantic schemas used to coax structured "reflection token"-style
judgments out of a general-purpose chat LLM (gpt-4o-mini has not been
fine-tuned with literal Self-RAG reflection tokens, so each judgment is
elicited via with_structured_output instead)."""

from typing import Literal

from pydantic import BaseModel, Field


class ResolvedQuestion(BaseModel):
    """Decide whether the current question is answerable on its own (after
    resolving any pronouns/references against conversation history) or is
    missing information even the history can't supply."""

    needs_clarification: bool = Field(
        description=(
            "True if the question can't be understood/answered even "
            "considering conversation history -- e.g. a pronoun ('她'/'他'/"
            "'这个从者') with no prior turn establishing who it refers to, "
            "or the question is missing an essential entity/scope."
        )
    )
    clarification_question: str = Field(
        default="",
        description=(
            "If needs_clarification, a short natural-language question asking "
            "the user for the missing information (same language as the "
            "original question). Empty string otherwise."
        ),
    )
    resolved_question: str = Field(
        default="",
        description=(
            "If NOT needs_clarification, the question rewritten to be fully "
            "self-contained, using conversation history to resolve pronouns/"
            "references (e.g. '她的宝具是什么' -> '阿尔托莉雅的宝具是什么' if "
            "阿尔托莉雅 was the servant just discussed). If the question is "
            "already self-contained, return it unchanged. Empty string if "
            "needs_clarification."
        ),
    )


class ConflictCheck(BaseModel):
    """Decide whether the retrieved documents disagree with each other on
    the specific fact the question is asking about (e.g. different noble
    phantasm ranks across servant variants that share a name) such that
    answering confidently from all of them at once would blend or average
    over a real distinction the user needs to pick between."""

    has_conflict: bool = Field(
        description=(
            "True if two or more of the documents give mutually inconsistent "
            "information that's directly relevant to answering the question "
            "(not just different documents covering different topics)."
        )
    )
    clarification_question: str = Field(
        default="",
        description=(
            "If has_conflict, a short question (same language as the "
            "original question) that names the conflicting values/sources "
            "and asks the user which one they mean. Empty string otherwise."
        ),
    )


class RouteQuery(BaseModel):
    """Decide whether a question should be answered from the structured
    servant database (exact game-mechanic facts) or the lore vector store
    (background story / personality / relationships)."""

    route: Literal["structured", "vectorstore"] = Field(
        description=(
            "'structured' for factual game-mechanic questions (skills, noble "
            "phantasm rank/card type, rarity, class, acquisition method); "
            "'vectorstore' for lore / background story / personality / "
            "relationship questions."
        )
    )
    servant_name: str = Field(
        description="The servant's Chinese name mentioned in the question, extracted verbatim as written."
    )
    class_hint: str = Field(
        default="",
        description=(
            "The English class name in lowercase (e.g. 'lancer') ONLY if a class/job "
            "(Saber, Lancer, Archer, Rider, Caster, Assassin, Berserker, Ruler, Alter "
            "Ego, Moon Cancer, etc.) is explicitly written in the question text itself. "
            "Do not infer or guess a class from general knowledge about the servant when "
            "the question does not mention one -- leave this an empty string in that case."
        ),
    )


class SubQuestionPlan(BaseModel):
    question: str = Field(description="A self-contained single-hop sub-question.")
    query_type: Literal["standard", "enumerate"] = Field(
        description=(
            "'enumerate' if the sub-question asks to list/count ALL occurrences "
            "of an entity across the whole corpus (e.g. '列出所有...出场的剧情/"
            "章节标题', '一共出现在多少个剧情里') -- these need an exhaustive "
            "keyword scan over every record, not top-K similarity search, which "
            "would only surface a handful of passages matching this specific "
            "phrasing and silently miss most real occurrences. 'standard' for a "
            "normal fact/lore question answerable from a few best-matching "
            "passages."
        )
    )
    entity_name: str = Field(
        default="",
        description=(
            "For 'enumerate' type: the plain Chinese name exactly as written "
            "in the ORIGINAL question, extracted verbatim -- do not translate, "
            "annotate, or append an English/romanized name in parentheses "
            "(e.g. use '伊阿宋', never '伊阿宋（Jason）'). This is matched as an "
            "exact substring against corpus text, so anything added that isn't "
            "in the source text will silently match nothing. Empty string for "
            "'standard'."
        ),
    )


class DecomposeQuery(BaseModel):
    is_complex: bool = Field(
        description=(
            "True if the question asks about more than one servant, or "
            "requires combining multiple distinct facts that would need "
            "separate lookups (e.g. a comparison)."
        )
    )
    sub_questions: list[SubQuestionPlan] = Field(
        description=(
            "If is_complex, a list of self-contained single-hop sub-questions "
            "(each answerable independently) that together cover the original "
            "question, each with its own query_type. If not complex, a list "
            "containing just the original question (still classify its query_type)."
        )
    )


class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if every claim in the generated answer is grounded in / "
            "supported by the provided facts, 'no' if it contains claims not "
            "supported by the facts."
        )
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer actually addresses the question asked, else 'no'."
    )


class RewrittenQuery(BaseModel):
    better_question: str = Field(
        description=(
            "A rewritten version of the question optimized for retrieval "
            "(clearer entity names, less ambiguity, no typos), preserving "
            "the original intent and language."
        )
    )
