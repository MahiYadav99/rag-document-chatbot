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
from llm.response_generator import generate_rag_response, format_context
from memory.conversation_memory import ConversationMemory
from rag.prompt import PromptType
from rag.retriever import create_retriever
from vectorstore.vector_db import load_vector_database


print("=" * 60)
print("RESPONSE GENERATOR TEST")
print("=" * 60)

# Load embedding model
embedding_model = load_embedding_model(
    EMBEDDING_MODEL
)

# Load existing vector database
vector_db = load_vector_database(
    embedding_model,
    CHROMA_DB_PATH,
)

# Create retriever
retriever = create_retriever(vector_db)

# Retrieve relevant documents
documents = retriever.invoke(
    "What is machine learning?"
)

# Create Gemini client
gemini_client = GeminiClient(
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

# Create conversation memory
memory = ConversationMemory()

# Generate RAG response
response = generate_rag_response(
    question="What is Machine Learning?",
    documents=documents,
    conversation_history=memory.format_history(),
    prompt_type=PromptType.QA,
    gemini_client=gemini_client,
)

print("=" * 60)
print("RAG RESPONSE")
print("=" * 60)
print(response.answer)

# Display formatted context
context = format_context(documents)

print("=" * 70)
print("FORMATTED CONTEXT")
print("=" * 70)
print(context)