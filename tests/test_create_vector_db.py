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
from vectorstore.vector_db import create_vector_database


print("=" * 60)
print("VECTOR DATABASE CREATION TEST")
print("=" * 60)

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

print(f"Documents Loaded : {len(documents)}")
print(f"Chunks Created   : {len(chunks)}")
print(f"Vectors Stored   : {vector_db._collection.count()}")

print("\nDatabase created successfully!")