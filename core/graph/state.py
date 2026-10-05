from typing import Any, TypedDict


class FinancialAgentState(TypedDict, total=False):
    question: str
    dataset_id: str
    document_path: str
    df: Any

    guardrail_safe: bool
    guardrail_result: dict
    guardrail_reason: str
    guardrail_classifier: Any
    
    output_guardrail_safe: bool
    output_guardrail_result: dict
    output_guardrail_reason: str
    output_guardrail_classifier: Any
    reasoning_llm: Any
    
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
