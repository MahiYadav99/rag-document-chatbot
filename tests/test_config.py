import sys
from pathlib import Path

# Add the project root to Python's search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    GEMINI_API_KEY,
    UPLOAD_FOLDER,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

print("=" * 50)
print("CONFIGURATION TEST")
print("=" * 50)

print(f"Gemini API Key Loaded : {'Yes' if GEMINI_API_KEY else 'No'}")
print(f"Upload Folder         : {UPLOAD_FOLDER}")
print(f"Chroma DB Path        : {CHROMA_DB_PATH}")
print(f"Embedding Model       : {EMBEDDING_MODEL}")
print(f"Chunk Size            : {CHUNK_SIZE}")
print(f"Chunk Overlap         : {CHUNK_OVERLAP}")

print("\nAll configuration values loaded successfully!")