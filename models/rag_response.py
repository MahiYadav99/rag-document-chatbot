from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag.prompt import PromptType


@dataclass
class RAGResponse:
    """
    Standard response returned by the RAG pipeline.
    """

    answer: str

    retrieved_documents: list[Document]

    prompt_type: PromptType

    raw_response: Any