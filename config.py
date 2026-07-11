import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# Project Root Directory
# ==========================================================

# Absolute path to the project folder
BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# API Configuration
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini Configuration
GEMINI_MODEL = "gemini-2.5-flash"

# Ensure the Gemini API key exists
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file."
    )

# ==========================================================
# Project Paths
# ==========================================================

UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"

CHROMA_DB_PATH = BASE_DIR / "database" / "chroma_db"

# ==========================================================
# Embedding Model Configuration
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ==========================================================
# Text Chunking Configuration
# ==========================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200