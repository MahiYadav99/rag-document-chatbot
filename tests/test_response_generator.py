import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from embeddings.embedding_model import load_embedding_model
from llm.gemini_client import GeminiClient
from rag.prompt import PromptType
from llm.response_generator import generate_rag_response
from rag.retriever import create_retriever
from vectorstore.vector_db import load_vector_database

print("=" * 60)
print("RESPONSE GENERATOR TEST")
print("=" * 60)

embedding_model = load_embedding_model(
    EMBEDDING_MODEL
)

vector_db = load_vector_database(
    embedding_model,
    CHROMA_DB_PATH,
)

retriever = create_retriever(vector_db)

documents = retriever.invoke(
    "What is machine learning?"
)

client = GeminiClient(
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

selected_prompt = PromptType.QA

response = generate_rag_response(
    question="What is Machine Learning?",
    documents=documents,
    prompt_type=selected_prompt,
    gemini_client=client,
)

print("=" * 60)
print("RAG RESPONSE")
print("=" * 60)

print(response.answer)


from llm.response_generator import format_context

context = format_context(documents)

print("=" * 70)
print("FORMATTED CONTEXT")
print("=" * 70)
print(context)