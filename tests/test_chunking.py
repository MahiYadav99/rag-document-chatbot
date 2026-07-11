import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CHUNK_OVERLAP, CHUNK_SIZE
from loaders.pdf_loader import load_pdf
from rag.chunking import split_documents

pdf_path = "data/uploads/sample.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

print("=" * 60)
print("CHUNKING TEST")
print("=" * 60)

print(f"Original Pages : {len(documents)}")
print(f"Total Chunks   : {len(chunks)}")

print(f"First Chunk Length: {len(chunks[0].page_content)}")
print("\nLast Chunk Length:")
print(len(chunks[-1].page_content))
print("\nFirst Chunk:\n")
print(chunks[0].page_content[:500])

print("\nMetadata:\n")
print(chunks[0].metadata)