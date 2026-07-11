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
from memory.conversation_memory import ConversationMemory
from rag.pipeline import RAGPipeline
from vectorstore.vector_db import load_vector_database
from rag.retriever import create_retriever

print("=" * 60)
print("PIPELINE TEST")
print("=" * 60)

# Step 1: Load embedding model
embedding_model = load_embedding_model(
    EMBEDDING_MODEL
)

# Step 2: Load vector database
vector_db = load_vector_database(
    embedding_model,
    CHROMA_DB_PATH,
)

# Step 3: Create retriever
retriever = create_retriever(
    vector_db
)

# Step 4: Create Gemini client
gemini_client = GeminiClient(
    api_key=GEMINI_API_KEY,
    model_name=GEMINI_MODEL,
)

# Step 5: Create conversation memory
memory = ConversationMemory()
print(memory.max_messages)

# Step 6: Create pipeline
pipeline = RAGPipeline(
    retriever=retriever,
    gemini_client=gemini_client,
    memory=memory,
)

print("\nQuestion 1")
print("-" * 40)

response1 = pipeline.ask(
    "What is Machine Learning?"
)

print(response1.answer)

print("\nQuestion 2")
print("-" * 40)

response2 = pipeline.ask(
    "Who proposed it?"
)

print(response2.answer)

print("\nConversation History")
print("-" * 40)

print(
    pipeline.memory.format_history()
)