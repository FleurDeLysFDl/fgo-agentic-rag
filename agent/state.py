from typing import TypedDict

# After this many consecutive turns where the assistant only asked a
# clarifying question (never landed on a real answer), resolve_question and
# check_conflict stop asking and commit to their best-effort interpretation
# instead. Without this, open-ended/narrative questions ("她和他的关系"-style)
# have no natural stopping point -- each answer can always be narrowed
# further, so the two clarification gates can keep firing indefinitely with
# the user never getting an actual answer (observed: 8 rounds of narrowing
# "情感" -> "情感深度" -> "谁对谁的情感" with no answer ever generated).
#
# Set to 1 (force resolution starting on the very next turn after any single
# clarification), not higher: prompting the model to recognize "the user's
# short reply already answers my own clarifying question" and resolve
# instead of asking again was tried and is unreliable (observed: gpt-4o-mini
# re-asked an almost word-for-word identical question after the user
# answered "所有" to "全部版本还是特定版本？"). A round-limit that only
# kicks in on round 2+ still lets one redundant repeat like that through
# every time; capping at 1 removes the model's ability to ask a second
# clarifying question at all, trading "occasionally answers a slightly
# broader question than intended" for "never loops or repeats itself."
MAX_CLARIFICATION_ROUNDS = 1


class Document(TypedDict):
    text: str
    source: str


class Turn(TypedDict):
    role: str  # "user" or "assistant"
    content: str


class SubQuestionPlan(TypedDict):
    question: str
    query_type: str  # "standard" or "enumerate"
    entity_name: str


class SubState(TypedDict):
    question: str
    route: str
    servant_name: str
    class_hint: str
    documents: list[Document]
    generation: str
    retrieve_retries: int
    generate_retries: int
    needs_clarification: bool
    clarification_question: str
    clarification_rounds: int


class GraphState(TypedDict):
    question: str
    history: list[Turn]
    clarification_rounds: int
    needs_clarification: bool
    clarification_question: str
    sub_questions: list[str]
    sub_question_plans: list[SubQuestionPlan]
    sub_answers: list[str]
    sub_documents: list[list[Document]]
    final_answer: str
