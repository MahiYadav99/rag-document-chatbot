from enum import Enum
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate


# ==========================================================
# Prompt Types
# ==========================================================

class PromptType(str, Enum):
    """
    Supported prompt templates.
    """

    QA = "qa"
    SUMMARY = "summary"
    ACADEMIC = "academic"


# ==========================================================
# Prompt Constants
# ==========================================================

QA_SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions ONLY using the provided document context.

Rules:
- Answer ONLY using the supplied context.
- Do NOT invent or assume information.
- If the answer cannot be found in the provided context, reply exactly:
  "I could not find the answer in the uploaded document."
- Keep your answer clear, concise, and well-structured.
- If the retrieved context contains page numbers or source information,
  cite them whenever possible.
"""

QA_HUMAN_PROMPT = """
Context:
{context}

Question:
{question}
"""


SUMMARY_SYSTEM_PROMPT = """
You are an expert document summarization assistant.

Generate a concise and accurate summary using ONLY the provided context.

Do not add information that is not present in the document.
"""

SUMMARY_HUMAN_PROMPT = """
Context:
{context}
"""


ACADEMIC_SYSTEM_PROMPT = """
You are an academic assistant.

Answer in a formal educational style.

Use ONLY the provided context.

If the answer cannot be found in the document,
clearly state that the information is unavailable.
"""

ACADEMIC_HUMAN_PROMPT = """
Context:
{context}

Question:
{question}
"""


# ==========================================================
# Prompt Builders
# ==========================================================

def create_qa_prompt() -> ChatPromptTemplate:
    """
    Create the Question-Answer prompt template.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("human", QA_HUMAN_PROMPT),
        ]
    )


def create_summary_prompt() -> ChatPromptTemplate:
    """
    Create the document summarization prompt template.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", SUMMARY_HUMAN_PROMPT),
        ]
    )


def create_academic_prompt() -> ChatPromptTemplate:
    """
    Create the academic explanation prompt template.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", ACADEMIC_SYSTEM_PROMPT),
            ("human", ACADEMIC_HUMAN_PROMPT),
        ]
    )


# ==========================================================
# Prompt Registry
# ==========================================================

PROMPT_REGISTRY: dict[
    PromptType,
    Callable[[], ChatPromptTemplate],
] = {
    PromptType.QA: create_qa_prompt,
    PromptType.SUMMARY: create_summary_prompt,
    PromptType.ACADEMIC: create_academic_prompt,
}


# ==========================================================
# Public API
# ==========================================================

def get_prompt(prompt_type: PromptType) -> ChatPromptTemplate:
    """
    Retrieve a prompt template by its type.

    Args:
        prompt_type (PromptType):
            The prompt template to retrieve.

    Returns:
        ChatPromptTemplate:
            The requested prompt template.

    Raises:
        ValueError:
            If the prompt type is unsupported.
    """

    builder = PROMPT_REGISTRY.get(prompt_type)

    if builder is None:
        raise ValueError(
            f"Unsupported prompt type: {prompt_type}"
        )

    return builder()