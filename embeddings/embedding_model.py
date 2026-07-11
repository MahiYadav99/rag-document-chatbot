from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Load and return a Hugging Face embedding model.

    Args:
        model_name (str):
            Name of the embedding model.

    Returns:
        HuggingFaceEmbeddings:
            Loaded embedding model.

    Raises:
        RuntimeError:
            If the model cannot be loaded.
    """

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name
        )

        return embedding_model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model: {e}"
        ) from e