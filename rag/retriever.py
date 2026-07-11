from langchain_chroma import Chroma


def create_retriever(
    vector_db: Chroma,
    k: int = 4,
    search_type: str = "similarity",
):
    """
    Create a retriever from a Chroma vector database.

    Args:
        vector_db (Chroma):
            Initialized Chroma vector database.

        k (int):
            Number of documents to retrieve.

        search_type (str):
            Retrieval strategy.
            Options:
                - "similarity"
                - "mmr"

    Returns:
        VectorStoreRetriever
    """

    return vector_db.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
        },
    )