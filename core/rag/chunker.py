class FinancialDocumentChunker:
    """Split financial documents into overlapping text chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[dict]) -> list[dict]:
        chunks = []

        for document in documents:
            text = document.get("text", "").strip()
            metadata = document.get("metadata", {}).copy()

            if not text:
                continue

            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = len(chunks)

                    chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": chunk_metadata,
                        }
                    )

                if end >= text_length:
                    break

                start = end - self.chunk_overlap

        return chunks
