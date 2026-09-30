from core.graph.state import FinancialAgentState
from core.engines.anomaly_detector import AnomalyDetector
from core.engines.financial_analyzer import FinancialAnalyzer
from core.engines.time_analyzer import TimeAnalyzer
from core.rag.rag_node import FinancialRAG
from core.mcp.client import MCPClient

financial_analyzer = FinancialAnalyzer()
time_analyzer = TimeAnalyzer()
anomaly_detector = AnomalyDetector()

mcp_client = MCPClient()

def mcp_market_node(state: FinancialAgentState):
    symbol = state.get("mcp_symbol")

    if not symbol:
        return {
            "result": [],
            "error": "No market symbol was provided for MCP."
        }

    result = mcp_client.get_market_quote(symbol)

    return {
        "result": result,
        "mcp_result": result,
    }

def financial_analysis_node(state):
    df = state["df"]
    metrics = state.get("metrics", {})

    result = financial_analyzer.run(
        df,
        metrics=metrics
    )

    return {
        "result": result
    }

financial_rag = FinancialRAG(
    chunk_size=500,
    chunk_overlap=50,
)

def rag_node(state):
    question = state["question"]
    document_path = state.get("document_path")

    if not document_path:
        return {
            "result": [],
            "error": "No financial document was provided for RAG."
        }

    financial_rag.index_document(document_path)

    results = financial_rag.retrieve(
        query=question,
        top_k=3,
    )

    return {
        "result": results
    }

def trend_node(state):
    df = state["df"]
    entities = state.get("entities", [])

    result = time_analyzer.run(
        df,
        entities=entities
    )

    return {
        "result": result
    }


def anomaly_node(state):
    df = state["df"]
    entities = state.get("entities", [])

    result = anomaly_detector.run(
        df,
        entities=entities
    )

    return {
        "result": result
    }

def evidence_fusion_node(state):
    result = state.get("result")

    if result is None:
        return {
            "evidence": []
        }

    evidence = {
        "source": state.get("route", "unknown"),
        "findings": result
    }

    return {
        "evidence": [evidence]
    }

def financial_reasoning_node(state: FinancialAgentState):
    evidence = state.get("evidence", [])

    if not evidence:
        return {
            "answer": "No financial evidence was produced."
        }

    findings = []

    for item in evidence:
        source = item.get("source", "unknown")
        source_findings = item.get("findings", [])

        findings.append(f"[{source}]")

        if source == "mcp" and isinstance(source_findings, dict):
            symbol = source_findings.get("symbol", "unknown")
            price = source_findings.get("price", "unknown")
            currency = source_findings.get("currency", "")
            change_percent = source_findings.get(
                "change_percent",
                "unknown",
            )
            source_name = source_findings.get(
                "source",
                "unknown",
            )
            timestamp = source_findings.get(
                "timestamp",
                "unknown",
            )

            findings.append(
                f"{symbol} market price: "
                f"{price} {currency}"
            )
            findings.append(
                f"Change: {change_percent}%"
            )
            findings.append(
                f"Source: {source_name}"
            )
            findings.append(
                f"Timestamp: {timestamp}"
            )

        elif isinstance(source_findings, list):
            for finding in source_findings:
                if isinstance(finding, dict):
                    text = finding.get("text", "")
                    metadata = finding.get("metadata", {})

                    source_name = metadata.get(
                        "source",
                        "unknown",
                    )
                    page = metadata.get(
                        "page",
                        "unknown",
                    )
                    score = finding.get("score")

                    if score is not None:
                        findings.append(
                            f"{text} "
                            f"(Source: {source_name}, "
                            f"Page: {page}, "
                            f"Score: {score:.3f})"
                        )
                    else:
                        findings.append(
                            f"{text} "
                            f"(Source: {source_name}, "
                            f"Page: {page})"
                        )
                else:
                    findings.append(str(finding))

        else:
            findings.append(str(source_findings))

    answer = "\n".join(findings)

    return {
        "answer": answer
    }
