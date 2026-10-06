from core.graph.state import FinancialAgentState
from core.engines.anomaly_detector import AnomalyDetector
from core.engines.financial_analyzer import FinancialAnalyzer
from core.engines.time_analyzer import TimeAnalyzer
from core.rag.rag_node import FinancialRAG
from core.mcp.client import MCPClient
from core.guardrails.input_guardrails import validate_input
from core.guardrails.classifier import GuardrailClassifier
from core.guardrails.output_classifier import OutputGuardrailClassifier
from core.observability.logging import get_logger, log_event
import logging
import time

financial_analyzer = FinancialAnalyzer()
time_analyzer = TimeAnalyzer()
anomaly_detector = AnomalyDetector()

mcp_client = MCPClient()

logger = get_logger(__name__)

def guardrail_node(state: FinancialAgentState):
    question = state.get("question")

    validation = validate_input(question)

    if not validation["valid"]:
        log_event(
            logger,
            logging.WARNING,
            "input_guardrail_decision",
            safe=False,
            validation_failed=True,
            reason=validation["reason"],
        )

        return {
            "guardrail_safe": False,
            "guardrail_reason": validation["reason"],
            "error": validation["reason"],
        }

    classifier = state.get("guardrail_classifier")

    if classifier is None:
        log_event(
            logger,
            logging.ERROR,
            "output_guardrail_decision",
            safe=False,
            classifier_configured=False,
        )

        return {
            "guardrail_safe": False,
            "guardrail_reason": "Guardrail classifier is not configured.",
            "error": "Guardrail classifier is not configured.",
        }

    result = classifier.classify(validation["query"])

    log_event(
        logger,
        logging.INFO if result.is_safe else logging.WARNING,
        "input_guardrail_decision",
        safe=result.is_safe,
        prompt_injection=result.is_prompt_injection,
        pii=result.is_pii,
        irrelevant=result.is_irrelevant,
    )

    if not result.is_safe:
        return {
            "guardrail_safe": False,
            "guardrail_result": result.model_dump(),
            "guardrail_reason": result.reason,
            "error": f"Request rejected by guardrail: {result.reason}",
        }

    return {
        "guardrail_safe": True,
        "guardrail_result": result.model_dump(),
        "guardrail_reason": result.reason,
        "question": validation["query"],
    }

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
        return {"answer": "No financial evidence was produced."}

    reasoning_llm = state.get("reasoning_llm")

    if reasoning_llm is not None:
        try:
            answer = reasoning_llm.reason(
                question=state["question"],
                evidence=evidence,
            )

            return {"answer": answer}

        except Exception as exc:
            return {
                "answer": (
                    "Financial reasoning failed. "
                    f"Reason: {exc}"
                )
            }

    # Fallback for tests and environments where no reasoning model
    # has been injected.
    findings = []

    for item in evidence:
        source = item.get("source", "unknown")
        source_findings = item.get("findings", [])

        findings.append(f"[{source}]")

        if source == "mcp" and isinstance(source_findings, dict):
            symbol = source_findings.get("symbol", "unknown")
            price = source_findings.get("price", "unknown")
            currency = source_findings.get("currency", "")
            change_percent = source_findings.get("change_percent", "unknown")

            findings.append(
                f"{symbol} market price: {price} {currency}"
            )
            findings.append(
                f"Change: {change_percent}%"
            )

        elif isinstance(source_findings, list):
            for finding in source_findings:
                if isinstance(finding, dict):
                    text = finding.get("text", "")
                    metadata = finding.get("metadata", {})

                    source_name = metadata.get("source", "unknown")
                    page = metadata.get("page", "unknown")
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
                            f"(Source: {source_name}, Page: {page})"
                        )

                else:
                    findings.append(str(finding))

        else:
            findings.append(str(source_findings))

    return {"answer": "\n".join(findings)}


def output_guardrail_node(state: FinancialAgentState):
    question = state.get("question", "")
    answer = state.get("answer", "")
    evidence = state.get("evidence", [])

    classifier = state.get("output_guardrail_classifier")

    if classifier is None:
        return {
            "output_guardrail_safe": False,
            "output_guardrail_reason": (
                "Output guardrail classifier is not configured."
            ),
            "error": "Output guardrail classifier is not configured.",
        }

    result = classifier.classify(
        question=question,
        answer=answer,
        evidence=evidence,
    )

    if not result.is_safe:
        log_event(
            logger,
            logging.INFO,
            "output_guardrail_decision",
            safe=True,
            grounded=result.is_grounded,
            unsupported_claims=result.contains_unsupported_claims,
            pii=result.contains_pii,
        )

        return {
            "output_guardrail_safe": False,
            "output_guardrail_result": result.model_dump(),
            "output_guardrail_reason": result.reason,
            "error": f"Answer rejected by output guardrail: {result.reason}",
        }

    return {
        "output_guardrail_safe": True,
        "output_guardrail_result": result.model_dump(),
        "output_guardrail_reason": result.reason,
    }
