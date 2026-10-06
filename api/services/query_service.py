import os

from dotenv import load_dotenv

from api.services.dataset_service import DatasetService

from core.agent.financial_reasoner import FinancialReasoner
from core.agent.llm_interface import LLMInterface
from core.graph.graph import build_graph
from core.guardrails.classifier import GuardrailClassifier
from core.guardrails.output_classifier import OutputGuardrailClassifier


load_dotenv()


class QueryService:
    def __init__(self, dataset_service: DatasetService):
        self.dataset_service = dataset_service

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the Financial Intelligence API."
            )

        llm = LLMInterface(
            api_key=api_key,
            model="gpt-4o-mini",
        )

        self.guardrail_classifier = GuardrailClassifier(llm)
        self.output_guardrail_classifier = OutputGuardrailClassifier(llm)
        self.reasoner = FinancialReasoner(llm)

        self.graph = build_graph()

    def answer_query(self, dataset_id: str, question: str) -> dict:
        df = self.dataset_service.get_dataset(dataset_id)

        state = {
            "question": question,
            "dataset_id": dataset_id,
            "df": df,
            "guardrail_classifier": self.guardrail_classifier,
            "output_guardrail_classifier": self.output_guardrail_classifier,
            "reasoning_llm": self.reasoner,
        }

        result = self.graph.invoke(state)

        if not result.get("guardrail_safe", False):
            raise ValueError(
                result.get(
                    "guardrail_reason",
                    "Query rejected by input guardrail.",
                )
            )

        if not result.get("output_guardrail_safe", False):
            raise ValueError(
                result.get(
                    "output_guardrail_reason",
                    "Answer rejected by output guardrail.",
                )
            )

        return {
            "answer": result.get("answer"),
            "result": result.get("result"),
            "evidence": result.get("evidence", []),
            "intent": result.get("intent"),
            "route": result.get("route"),
        }
