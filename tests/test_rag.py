from core.rag.rag_node import FinancialRAG


def test_financial_rag_retrieval():
    rag = FinancialRAG(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = rag.index_document(
        "data/documents/test_financial_report.pdf"
    )

    assert chunks > 0

    results = rag.retrieve(
        "What happened to operating profit?",
        top_k=1,
    )

    assert len(results) == 1
    assert "operating profit" in results[0]["text"].lower()
    assert results[0]["metadata"]["source"] == "test_financial_report.pdf"
    assert results[0]["metadata"]["page"] == 1
