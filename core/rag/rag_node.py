from core.rag.chunker import FinancialDocumentChunker
from core.rag.document_loader import FinancialDocumentLoader
from core.rag.embeddings import FinancialEmbeddingModel
from core.rag.retriever import FinancialRetriever
from core.rag.vector_store import FinancialVectorStore


class FinancialRAG:
    """Financial document retrieval pipeline."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.loader = FinancialDocumentLoader()

        self.chunker = FinancialDocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_model = FinancialEmbeddingModel()

        self.vector_store = FinancialVectorStore()

        self.retriever = FinancialRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

    def index_document(self, file_path: str) -> int:
        documents = self.loader.load(file_path)

        chunks = self.chunker.chunk(documents)

        if not chunks:
            raise ValueError("No usable text was extracted from the document.")

        embeddings = self.embedding_model.embed_documents(
            [chunk["text"] for chunk in chunks]
        )

        self.vector_store.add(
            documents=chunks,
            embeddings=embeddings,
        )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )
