from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class FinancialDocumentLoader:
    """Load financial PDF documents with page-level metadata."""

    def load(self, file_path: str) -> list[dict]:
        """
        Extract text from a PDF page by page.

        Returns:
            A list of dictionaries containing page text and metadata.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Document path is not a file: {file_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF documents are supported."
            )

        reader = PdfReader(str(path))

        documents: list[dict] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to extract text from page "
                    f"{page_number} of '{path.name}'."
                ) from exc

            text = text.strip()

            # Skip pages with no extractable text.
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

        if not documents:
            raise ValueError(
                f"No extractable text found in PDF: {path}"
            )

        return documents