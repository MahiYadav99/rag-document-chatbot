import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
)

from embeddings.embedding_model import load_embedding_model
from vectorstore.vector_db import load_vector_database


print("=" * 60)
print("VECTOR DATABASE LOAD TEST")
print("=" * 60)

embedding_model = load_embedding_model(
    EMBEDDING_MODEL
)

vector_db = load_vector_database(
    embedding_model,
    CHROMA_DB_PATH,
)

print(f"Vectors Stored : {vector_db._collection.count()}")

print("\nExisting vector database loaded successfully!")