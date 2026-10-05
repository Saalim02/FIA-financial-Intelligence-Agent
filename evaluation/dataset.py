EVALUATION_DATASET = [
    {
        "id": "financial_001",
        "question": "What was the revenue?",
        "expected_route": "financial",
        "category": "financial_analysis",
    },
    {
        "id": "trend_001",
        "question": "What is the revenue trend over time?",
        "expected_route": "trend",
        "category": "trend_analysis",
    },
    {
        "id": "anomaly_001",
        "question": "Are there any unusual revenue values?",
        "expected_route": "anomaly",
        "category": "anomaly_detection",
        "reference": "The annual report says that revenue increased from 100 million to 120 million, which represents 20% revenue growth.",
    },
    {
        "id": "rag_001",
        "question": "What does the annual report say about revenue growth?",
        "expected_route": "rag",
        "category": "document_question",
        "reference": "The annual report says that revenue increased from 100 million to 120 million, which represents 20% revenue growth.",
    },
    {
        "id": "mcp_001",
        "question": "What is the current stock price?",
        "expected_route": "mcp",
        "category": "market_data",
    },
]
