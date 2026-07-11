import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import EMBEDDING_MODEL
from embeddings.embedding_model import load_embedding_model

embedding_model = load_embedding_model(EMBEDDING_MODEL)

sample_text = "Artificial Intelligence is transforming healthcare."

embedding = embedding_model.embed_query(sample_text)

print("=" * 60)
print("EMBEDDING MODEL TEST")
print("=" * 60)

print(f"Model Loaded Successfully")

print(f"\nEmbedding Dimension : {len(embedding)}")

print("\nFirst 10 Values:")

print(embedding[:10])