import os
from dotenv import load_dotenv
from langsmith import traceable

from api.services.dataset_service import DatasetService

from core.agent.financial_reasoner import FinancialReasoner
from core.agent.llm_interface import LLMInterface
from core.agent.session_store import InMemorySessionStore
from core.graph.graph import build_graph
from core.guardrails.classifier import GuardrailClassifier
from core.guardrails.output_classifier import OutputGuardrailClassifier


_UNSET = object()


load_dotenv()


class QueryService:
    def __init__(
        self,
        dataset_service: DatasetService,
        *,
        guardrail_classifier=_UNSET,
        output_guardrail_classifier=_UNSET,
        reasoner=_UNSET,
        graph=None,
        session_store=None,
    ):
        self.dataset_service = dataset_service

        injected_components = (guardrail_classifier, output_guardrail_classifier, reasoner)
        if any(component is _UNSET for component in injected_components):
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required for the Financial Intelligence API."
                )

            llm = LLMInterface(api_key=api_key, model="gpt-4o-mini")
            if guardrail_classifier is _UNSET:
                guardrail_classifier = GuardrailClassifier(llm)
            if output_guardrail_classifier is _UNSET:
                output_guardrail_classifier = OutputGuardrailClassifier(llm)
            if reasoner is _UNSET:
                if os.getenv("FINANCIAL_REASONING_PROVIDER", "qwen").strip().lower() == "qwen":
                    from core.agent.qwen_reasoning_provider import QwenReasoningProvider
                    reasoner = FinancialReasoner(QwenReasoningProvider())
                else:
                    reasoner = FinancialReasoner(llm)

        self.guardrail_classifier = guardrail_classifier
        self.output_guardrail_classifier = output_guardrail_classifier
        self.reasoner = reasoner

        self.graph = graph or build_graph()
        self.session_store = session_store or InMemorySessionStore()

    def model_status(self) -> dict:
        provider = getattr(self.reasoner, "llm", None)
        if provider is not None and hasattr(provider, "status"):
            return provider.status()
        return {"provider": "openai", "model": getattr(provider, "model", "unknown"),
                "adapter_loaded": False, "quantized": False, "device": "remote"}

    @traceable(name="financial_query")
    def answer_query(
        self,
        dataset_id: str,
        question: str,
        session_id: str | None = None,
        document_path: str | None = None,
    ) -> dict:
        df = self.dataset_service.get_dataset(dataset_id)

        conversation_history = (
            self.session_store.history_for(session_id, dataset_id)
            if session_id else []
        )

        state = {
            "question": question,
            "dataset_id": dataset_id,
            "df": df,
            "document_path": document_path,
            "guardrail_classifier": self.guardrail_classifier,
            "output_guardrail_classifier": self.output_guardrail_classifier,
            "reasoning_llm": self.reasoner,
            "conversation_history": conversation_history,
        }

        result = self.graph.invoke(state, config={"run_name": "financial_agent_graph"})

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

        response = {
            "answer": result.get("answer"),
            "result": result.get("result"),
            "evidence": result.get("evidence", []),
            "intent": result.get("intent"),
            "route": result.get("route"),
            "visualization": result.get("visualization"),
        }

        if session_id:
            self.session_store.append(
                session_id,
                dataset_id,
                question,
                response.get("answer"),
            )

        return response
