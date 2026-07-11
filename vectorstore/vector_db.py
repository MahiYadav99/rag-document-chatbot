from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_vector_database(
    documents: list[Document],
    embedding_model: Embeddings,
    database_path: str | Path,
) -> Chroma:
    """
    Create and persist a Chroma vector database.
    """

    if not documents:
        raise ValueError("No documents were provided.")

    database_path = Path(database_path)
    database_path.mkdir(parents=True, exist_ok=True)

    try:

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(database_path),
        )

        return vector_db

    except Exception as e:
        raise RuntimeError(
            f"Failed to create vector database: {e}"
        ) from e


def load_vector_database(
    embedding_model: Embeddings,
    database_path: str | Path,
) -> Chroma:
    """
    Load an existing Chroma vector database.
    """

    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Vector database not found at: {database_path}"
        )

    try:

        vector_db = Chroma(
            persist_directory=str(database_path),
            embedding_function=embedding_model,
        )

        return vector_db

    except Exception as e:
        raise RuntimeError(
            f"Failed to load vector database: {e}"
        ) from e