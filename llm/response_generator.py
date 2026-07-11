from langchain_core.documents import Document
from models.rag_response import RAGResponse
from memory.conversation_memory import ConversationMemory
from pathlib import Path


from llm.gemini_client import GeminiClient
from rag.prompt import (
    PromptType,
    get_prompt,
)


def format_context(
    documents: list[Document],
) -> str:
    """
    Convert retrieved LangChain documents into a structured
    context string for the language model.
    """

    if not documents:
        return "No relevant context was retrieved."

    formatted_chunks = []

    for index, document in enumerate(documents, start=1):

        source = Path(
            document.metadata.get(
            "source",
            "Unknown Source",
            )
        ).name

        page = document.metadata.get(
            "page_label",
            document.metadata.get("page", "Unknown"),
        )

        chunk = f"""
============================================================
DOCUMENT {index}
============================================================

Document Source:
{source}

Page Number:
{page}

Content:
------------------------------------------------------------
{document.page_content}

"""

        formatted_chunks.append(chunk.strip())

    return "\n\n".join(formatted_chunks)


def generate_rag_response(
    question: str,
    documents: list[Document],
    prompt_type: PromptType,
    gemini_client: GeminiClient,
    conversation_history: ConversationMemory,
):
    """
    Generate a response using the RAG pipeline.

    Args:
        question:
            User question.

        documents:
            Retrieved documents.

        prompt_type:
            Prompt template to use.

        gemini_client:
            Initialized Gemini client.

    Returns:
        GenerateContentResponse
    """

    context = format_context(documents)

    prompt_template = get_prompt(prompt_type)

    prompt = prompt_template.invoke(
    {
        "conversation_history": conversation_history,
        "context": context,
        "question": question,
    }
)

    formatted_prompt = "\n\n".join(
        message.content
        for message in prompt.messages
    )

    response = gemini_client.generate_response(
        formatted_prompt
    )

    return RAGResponse(
    answer=response.text,
    retrieved_documents=documents,
    prompt_type=prompt_type,
    raw_response=response,
)