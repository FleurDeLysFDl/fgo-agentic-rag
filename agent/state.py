from typing import TypedDict


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


class GraphState(TypedDict):
    question: str
    history: list[Turn]
    needs_clarification: bool
    clarification_question: str
    sub_questions: list[str]
    sub_answers: list[str]
    sub_documents: list[list[Document]]
    final_answer: str
