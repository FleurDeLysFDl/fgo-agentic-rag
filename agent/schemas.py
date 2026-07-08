"""Pydantic schemas used to coax structured "reflection token"-style
judgments out of a general-purpose chat LLM (gpt-4o-mini has not been
fine-tuned with literal Self-RAG reflection tokens, so each judgment is
elicited via with_structured_output instead)."""

from typing import Literal

from pydantic import BaseModel, Field


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


class DecomposeQuery(BaseModel):
    is_complex: bool = Field(
        description=(
            "True if the question asks about more than one servant, or "
            "requires combining multiple distinct facts that would need "
            "separate lookups (e.g. a comparison)."
        )
    )
    sub_questions: list[str] = Field(
        description=(
            "If is_complex, a list of self-contained single-hop sub-questions "
            "(each answerable independently) that together cover the original "
            "question. If not complex, a list containing just the original question."
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
