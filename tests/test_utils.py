from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)

from loaders.pdf_loader import load_pdf
from rag.chunking import split_documents
from embeddings.embedding_model import load_embedding_model


def prepare_documents(pdf_path: str):
    """Load and split a PDF into chunks."""

    documents = load_pdf(pdf_path)

    chunks = split_documents(
        documents,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    return documents, chunks


def prepare_embedding_model():
    """Load the embedding model."""

    return load_embedding_model(
        EMBEDDING_MODEL
    )