from core.graph.state import FinancialAgentState
from core.engines.anomaly_detector import AnomalyDetector
from core.engines.financial_analyzer import FinancialAnalyzer
from core.engines.time_analyzer import TimeAnalyzer
from core.rag.rag_node import FinancialRAG

financial_analyzer = FinancialAnalyzer()
time_analyzer = TimeAnalyzer()
anomaly_detector = AnomalyDetector()


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

        if isinstance(source_findings, list):
            for finding in source_findings:
                if isinstance(finding, dict):
                    text = finding.get("text", "")
                    metadata = finding.get("metadata", {})

                    source_name = metadata.get("source", "unknown")
                    page = metadata.get("page", "unknown")
                    score = finding.get("score")

                    findings.append(
                        f"{text} "
                        f"(Source: {source_name}, "
                        f"Page: {page}, "
                        f"Score: {score:.3f})"
                    )
                else:
                    findings.append(str(finding))
        else:
            findings.append(str(source_findings))

    answer = "\n".join(findings)

    return {
        "answer": answer
    }