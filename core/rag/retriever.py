from core.rag.embeddings import FinancialEmbeddingModel
from core.rag.vector_store import FinancialVectorStore


class FinancialRetriever:
    """Retrieve relevant financial document chunks."""

    def __init__(
        self,
        embedding_model: FinancialEmbeddingModel,
        vector_store: FinancialVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
