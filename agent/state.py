from typing import TypedDict

# After this many consecutive turns where the assistant only asked a
# clarifying question (never landed on a real answer), resolve_question and
# check_conflict stop asking and commit to their best-effort interpretation
# instead. Without this, open-ended/narrative questions ("她和他的关系"-style)
# have no natural stopping point -- each answer can always be narrowed
# further, so the two clarification gates can keep firing indefinitely with
# the user never getting an actual answer (observed: 8 rounds of narrowing
# "情感" -> "情感深度" -> "谁对谁的情感" with no answer ever generated).
MAX_CLARIFICATION_ROUNDS = 2


class Document(TypedDict):
    text: str
    source: str


class Turn(TypedDict):
    role: str  # "user" or "assistant"
    content: str


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
    sub_answers: list[str]
    sub_documents: list[list[Document]]
    final_answer: str
