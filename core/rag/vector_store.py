import numpy as np


class FinancialVectorStore:
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add(
        self,
        documents: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings."
            )

        if not documents:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if self.embeddings is None:
            self.embeddings = vectors
        else:
            if self.embeddings.shape[1] != vectors.shape[1]:
                raise ValueError("Embedding dimensions do not match.")

            self.embeddings = np.vstack(
                [self.embeddings, vectors]
            )

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[dict]:
        if self.embeddings is None or not self.documents:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")

        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match.")

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        document_norms = np.linalg.norm(
            self.embeddings,
            axis=1,
        )

        similarities = (
            self.embeddings @ query_vector
        ) / (
            document_norms * query_norm
        )

        ranked_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[index].copy()
            document["score"] = float(similarities[index])
            results.append(document)

        return results
