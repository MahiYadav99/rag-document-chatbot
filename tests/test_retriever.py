import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)

from embeddings.embedding_model import load_embedding_model
from loaders.pdf_loader import load_pdf
from rag.chunking import split_documents
from rag.retriever import create_retriever
from vectorstore.vector_db import create_vector_database

# Load PDF
documents = load_pdf("data/uploads/sample.pdf")

# Split into chunks
chunks = split_documents(
    documents,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Load embedding model
embedding_model = load_embedding_model(
    EMBEDDING_MODEL
)

# Create vector database
vector_db = create_vector_database(
    chunks,
    embedding_model,
    CHROMA_DB_PATH,
)

# Create retriever
retriever = create_retriever(vector_db)

# Ask a question
query = "What is machine learning?"

results = retriever.invoke(query)

print("=" * 60)
print("RETRIEVER TEST")
print("=" * 60)

print(f"Retrieved Chunks: {len(results)}")

for i, doc in enumerate(results, start=1):
    print(f"\n{'=' * 20} Chunk {i} {'=' * 20}")
    print(f"Page: {doc.metadata.get('page_label')}")
    print(f"Source: {doc.metadata.get('source')}")
    print("\nContent Preview:\n")
    print(doc.page_content[:400])