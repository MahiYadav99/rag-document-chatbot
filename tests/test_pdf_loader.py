import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from loaders.pdf_loader import load_pdf


pdf_path = "data/uploads/sample.pdf"

documents = load_pdf(pdf_path)

print("=" * 60)
print("PDF LOADER TEST")
print("=" * 60)

print(f"Number of pages : {len(documents)}")

print("\nFirst page preview:\n")

print(documents[0].page_content[:500])

print("\nMetadata:\n")

print(documents[0].metadata)