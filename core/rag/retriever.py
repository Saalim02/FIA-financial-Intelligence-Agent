import logging
import time

from core.observability.logging import get_logger, log_event
from core.rag.embeddings import FinancialEmbeddingModel
from core.rag.vector_store import FinancialVectorStore


logger = get_logger(__name__)


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

        start = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "rag_retrieval_started",
            query_length=len(query),
            top_k=top_k,
        )

        try:
            query_embedding = self.embedding_model.embed_query(query)

            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.INFO,
                "rag_retrieval_completed",
                query_length=len(query),
                top_k=top_k,
                result_count=len(results),
                latency_ms=latency_ms,
            )

            return results

        except Exception as exc:
            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.ERROR,
                "rag_retrieval_failed",
                query_length=len(query),
                top_k=top_k,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

            raise
