from langgraph.graph import END, START, StateGraph

from core.graph.nodes import (
    anomaly_node,
    evidence_fusion_node,
    financial_analysis_node,
    financial_reasoning_node,
    trend_node,
)
from core.graph.state import FinancialAgentState


def query_understanding(state: FinancialAgentState):
    question = state["question"].lower()

    if any(
        word in question
        for word in ["anomaly", "unusual", "outlier", "abnormal"]
    ):
        return {
            "intent": "anomaly_detection",
            "route": "anomaly",
        }

    if any(
        word in question
        for word in ["trend", "over time", "year", "growth", "change"]
    ):
        return {
            "intent": "trend_analysis",
            "route": "trend",
        }

    return {
        "intent": "financial_analysis",
        "route": "financial",
    }


def route_question(state: FinancialAgentState):
    return state["route"]


def build_graph():
    builder = StateGraph(FinancialAgentState)

    # Analysis nodes
    builder.add_node("query_understanding", query_understanding)
    builder.add_node("financial_analysis", financial_analysis_node)
    builder.add_node("trend_analysis", trend_node)
    builder.add_node("anomaly_detection", anomaly_node)
    builder.add_node("financial_reasoning", financial_reasoning_node)
    # Evidence fusion
    builder.add_node("evidence_fusion", evidence_fusion_node)

    # START -> Query Understanding
    builder.add_edge(START, "query_understanding")

    # Conditional routing
    builder.add_conditional_edges(
        "query_understanding",
        route_question,
        {
            "financial": "financial_analysis",
            "trend": "trend_analysis",
            "anomaly": "anomaly_detection",
        },
    )

    # Analysis -> Evidence Fusion
    builder.add_edge("financial_analysis", "evidence_fusion")
    builder.add_edge("trend_analysis", "evidence_fusion")
    builder.add_edge("anomaly_detection", "evidence_fusion")

    # Evidence Fusion -> END
    builder.add_edge("evidence_fusion", "financial_reasoning")
    builder.add_edge("financial_reasoning", END)

    return builder.compile()


financial_graph = build_graph()