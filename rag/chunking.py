from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Split LangChain Document objects into smaller chunks.

    Args:
        documents (list[Document]):
            List of documents to split.

        chunk_size (int):
            Maximum size of each chunk.

        chunk_overlap (int):
            Number of overlapping characters between chunks.

    Returns:
        list[Document]:
            List of chunked Document objects.

    Raises:
        ValueError:
            If the document list is empty.

        RuntimeError:
            If chunking fails.
    """

    if not documents:
        raise ValueError("No documents were provided for chunking.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks were created.")

        return chunks

    except Exception as e:
        raise RuntimeError(f"Failed to split documents: {e}") from e