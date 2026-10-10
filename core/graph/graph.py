import logging
import time
import re

from core.observability.logging import get_logger, log_event

from langgraph.graph import END, START, StateGraph

from core.data.column_resolver import ColumnResolver

from core.graph.nodes import (
    guardrail_node,
    anomaly_node,
    evidence_fusion_node,
    financial_analysis_node,
    financial_reasoning_node,
    mcp_market_node,
    output_guardrail_node,
    rag_node,
    trend_node,
)

from core.graph.state import FinancialAgentState


logger = get_logger(__name__)


def observe_node(node_name, node_function):
    """Wrap a LangGraph node with timing and error logging."""

    def wrapped(state):
        start = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "graph_node_started",
            node=node_name,
        )

        try:
            result = node_function(state)

            duration_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.INFO,
                "graph_node_completed",
                node=node_name,
                duration_ms=duration_ms,
            )

            return result

        except Exception as exc:
            duration_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.ERROR,
                "graph_node_failed",
                node=node_name,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )

            raise

    return wrapped

def extract_market_symbol(question: str) -> str | None:
    """
    Extract a stock/market ticker from a natural-language question.

    Examples:
        "What is the current stock price of AAPL?" -> "AAPL"
        "Give me the market quote for MSFT" -> "MSFT"

    This is intentionally generic and does not maintain
    a hardcoded list of companies.
    """
    matches = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", question)

    ignored = {
        "WHAT",
        "IS",
        "THE",
        "CURRENT",
        "STOCK",
        "PRICE",
        "SHARE",
        "MARKET",
        "QUOTE",
        "OF",
        "FOR",
        "GIVE",
        "ME",
    }

    for match in matches:
        if match not in ignored:
            return match

    return None


def query_understanding(state: FinancialAgentState):
    """
    Understand the user's question and dynamically resolve
    relevant dataset columns.

    This function does not assume any particular dataset schema.

    For financial questions, all explicitly requested dataset
    columns are preserved as metrics so multi-metric questions
    such as:

        "What was the total sales and profit?"

    can be answered using both Sales and Profit.
    """

    question = state["question"]
    question_lower = question.lower()
    # Resolve short follow-ups against this session's recent questions.
    history = state.get("conversation_history", [])
    contextual_question = question
    if history and any(term in question_lower for term in ("which one", "grew", "faster", "that", "those")):
        contextual_question = " ".join(
            [turn.get("question", "") for turn in history[-4:]] + [question]
        )

    df = state.get("df")

    # ---------------------------------------------------------
    # Resolve dataset columns dynamically
    # ---------------------------------------------------------

    entities = []
    numeric_column = None

    if df is not None:
        resolution = ColumnResolver.resolve(
            df=df,
            question=contextual_question,
        )

        entities = resolution.get(
            "explicit_columns",
            [],
        )

        numeric_column = resolution.get(
            "numeric_column",
        )

        # If the semantic resolver found a numeric column,
        # make sure it is available to downstream analysis.
        if numeric_column and numeric_column not in entities:
            entities.append(numeric_column)

    # ---------------------------------------------------------
    # MCP / market questions
    # ---------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "stock",
            "share price",
            "market price",
            "current price",
            "market",
            "quote",
        ]
    ):
        mcp_symbol = extract_market_symbol(question)

        return {
            "intent": "market_data",
            "route": "mcp",
            "entities": entities,
            "mcp_symbol": mcp_symbol,
        }

    # ---------------------------------------------------------
    # RAG / document questions
    # ---------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "report",
            "annual report",
            "document",
            "management",
            "according to",
            "according to the report",
        ]
    ):
        return {
            "intent": "document_analysis",
            "route": "rag",
            "entities": entities,
        }

    # ---------------------------------------------------------
    # Anomaly questions
    # ---------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "anomaly",
            "unusual",
            "outlier",
            "abnormal",
        ]
    ):
        return {
            "intent": "anomaly_detection",
            "route": "anomaly",
            "entities": entities,
        }

    # ---------------------------------------------------------
    # Trend questions
    # ---------------------------------------------------------

    if any(
        word in question_lower
        for word in [
            "trend",
            "over time",
            "year",
            "yearly",
            "annually",
            "annual",
            "quarterly",
            "quarter",
            "monthly",
            "month",
            "weekly",
            "week",
            "daily",
            "day",
            "growth",
            "change",
            "grew",
            "faster",
        ]
    ):
        return {
            "intent": "trend_analysis",
            "route": "trend",
            "entities": entities,
        }

    # ---------------------------------------------------------
    # Default financial analysis
    # ---------------------------------------------------------

    # Preserve every explicitly mentioned dataset column.
    #
    # Example:
    #
    # "What was the total sales and profit?"
    #
    # ColumnResolver -> ["Sales", "Profit"]
    #
    # Therefore:
    #
    # metrics = {
    #     "metrics": ["Sales", "Profit"]
    # }
    #
    # FinancialAnalyzer already supports this format.
    requested_metrics = [
        column
        for column in entities
        if df is not None and column in df.columns
    ]

    # Fallback to the semantic numeric resolver when no
    # explicit dataset column was mentioned.
    if not requested_metrics and numeric_column:
        requested_metrics = [numeric_column]

    return {
        "intent": "financial_analysis",
        "route": "financial",
        "entities": entities,
        "metrics": {
            "metrics": requested_metrics,
        },
    }


def route_question(state: FinancialAgentState):
    return state["route"]


def route_after_guardrail(state: FinancialAgentState):
    if state.get("guardrail_safe") is True:
        return "safe"

    return "unsafe"


def output_guardrail_rejection_node(state: FinancialAgentState):
    reason = state.get(
        "output_guardrail_reason",
        "The generated answer could not be safely returned.",
    )

    return {
        "output_guardrail_safe": False,
        "answer": (
            "The generated answer was blocked by the output guardrail: "
            f"{reason}"
        ),
    }


def route_after_output_guardrail(state: FinancialAgentState):
    if state.get("output_guardrail_safe") is True:
        return "safe"

    return "unsafe"


def guardrail_rejection_node(state: FinancialAgentState):
    reason = state.get(
        "guardrail_reason",
        "The request could not be processed.",
    )

    return {
        "answer": (
            "Request rejected by security guardrail: "
            f"{reason}"
        )
    }


def build_graph():
    builder = StateGraph(FinancialAgentState)

    # ---------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------

    builder.add_node(
        "query_understanding",
        observe_node(
            "query_understanding",
            query_understanding,
        ),
    )

    builder.add_node(
        "financial_analysis",
        observe_node(
            "financial_analysis",
            financial_analysis_node,
        ),
    )

    builder.add_node(
        "trend_analysis",
        observe_node(
            "trend_analysis",
            trend_node,
        ),
    )

    builder.add_node(
        "anomaly_detection",
        observe_node(
            "anomaly_detection",
            anomaly_node,
        ),
    )

    builder.add_node(
        "rag",
        observe_node(
            "rag",
            rag_node,
        ),
    )

    builder.add_node(
        "mcp",
        observe_node(
            "mcp",
            mcp_market_node,
        ),
    )

    builder.add_node(
        "evidence_fusion",
        observe_node(
            "evidence_fusion",
            evidence_fusion_node,
        ),
    )

    builder.add_node(
        "financial_reasoning",
        observe_node(
            "financial_reasoning",
            financial_reasoning_node,
        ),
    )

    builder.add_node(
        "guardrail",
        observe_node(
            "guardrail",
            guardrail_node,
        ),
    )

    builder.add_node(
        "guardrail_rejection",
        observe_node(
            "guardrail_rejection",
            guardrail_rejection_node,
        ),
    )

    builder.add_node(
        "output_guardrail",
        observe_node(
            "output_guardrail",
            output_guardrail_node,
        ),
    )

    builder.add_node(
        "output_guardrail_rejection",
        observe_node(
            "output_guardrail_rejection",
            output_guardrail_rejection_node,
        ),
    )

    # ---------------------------------------------------------
    # START -> Guardrail
    # ---------------------------------------------------------

    builder.add_edge(
        START,
        "guardrail",
    )

    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "safe": "query_understanding",
            "unsafe": "guardrail_rejection",
        },
    )

    builder.add_edge(
        "guardrail_rejection",
        END,
    )

    # ---------------------------------------------------------
    # Query Understanding -> Route
    # ---------------------------------------------------------

    builder.add_conditional_edges(
        "query_understanding",
        route_question,
        {
            "financial": "financial_analysis",
            "trend": "trend_analysis",
            "anomaly": "anomaly_detection",
            "rag": "rag",
            "mcp": "mcp",
        },
    )

    # ---------------------------------------------------------
    # Analysis -> Evidence Fusion
    # ---------------------------------------------------------

    builder.add_edge(
        "financial_analysis",
        "evidence_fusion",
    )

    builder.add_edge(
        "trend_analysis",
        "evidence_fusion",
    )

    builder.add_edge(
        "anomaly_detection",
        "evidence_fusion",
    )

    builder.add_edge(
        "rag",
        "evidence_fusion",
    )

    builder.add_edge(
        "mcp",
        "evidence_fusion",
    )

    # ---------------------------------------------------------
    # Evidence Fusion -> Reasoning
    # ---------------------------------------------------------

    builder.add_edge(
        "evidence_fusion",
        "financial_reasoning",
    )

    # ---------------------------------------------------------
    # Reasoning -> Output Guardrail
    # ---------------------------------------------------------

    builder.add_edge(
        "financial_reasoning",
        "output_guardrail",
    )

    builder.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {
            "safe": END,
            "unsafe": "output_guardrail_rejection",
        },
    )

    builder.add_edge(
        "output_guardrail_rejection",
        END,
    )

    return builder.compile()


financial_graph = build_graph()
