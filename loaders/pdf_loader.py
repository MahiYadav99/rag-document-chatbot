from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document



def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return its contents as a list of LangChain Document objects.

    Args:
        file_path (str):
            Path to the PDF file.

    Returns:
        list[Document]:
            A list of LangChain Document objects, where each Document represents one page of the PDF.

    Raises:
        FileNotFoundError:
            If the PDF file does not exist.

        ValueError:
            If the file is not a PDF.

        RuntimeError:
            If the PDF cannot be loaded.
    """

    pdf_path = Path(file_path)

    # Check if the file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Check file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        if not documents:
            raise ValueError("The PDF contains no readable content.")

        return documents

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {e}") from e
    
