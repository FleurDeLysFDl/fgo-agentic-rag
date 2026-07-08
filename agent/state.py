from typing import TypedDict


class Document(TypedDict):
    text: str
    source: str


class SubState(TypedDict):
    question: str
    route: str
    servant_name: str
    class_hint: str
    documents: list[Document]
    generation: str
    retrieve_retries: int
    generate_retries: int


class GraphState(TypedDict):
    question: str
    sub_questions: list[str]
    sub_answers: list[str]
    sub_documents: list[list[Document]]
    final_answer: str
