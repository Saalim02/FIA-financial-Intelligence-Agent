from typing import Any, TypedDict


class FinancialAgentState(TypedDict, total=False):
    question: str
    dataset_id: str
    document_path: str
    df: Any

    intent: str
    route: str

    metrics: dict
    entities: list[str]

    mcp_symbol: str
    mcp_result: dict

    result: Any
    evidence: list[dict]
    answer: str
    error: str
