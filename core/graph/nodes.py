from core.engines.anomaly_detector import AnomalyDetector
from core.engines.financial_analyzer import FinancialAnalyzer
from core.engines.time_analyzer import TimeAnalyzer


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
    
def financial_reasoning_node(state):
    evidence = state.get("evidence", [])

    if not evidence:
        return {
            "answer": "No financial evidence was produced."
        }

    findings = []

    for item in evidence:
        source = item.get("source", "unknown")
        source_findings = item.get("findings", [])

        findings.append(
            f"[{source}]"
        )

        if isinstance(source_findings, list):
            findings.extend(source_findings)
        else:
            findings.append(str(source_findings))

    answer = "\n".join(findings)

    return {
        "answer": answer
    }