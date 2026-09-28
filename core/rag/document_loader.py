from pathlib import Path

from pypdf import PdfReader


class FinancialDocumentLoader:
    """Load financial PDF documents with page-level metadata."""

    def load(self, file_path: str) -> list[dict]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF documents are supported.")

        reader = PdfReader(str(path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "page": page_number,
                    },
                }
            )

        return documents