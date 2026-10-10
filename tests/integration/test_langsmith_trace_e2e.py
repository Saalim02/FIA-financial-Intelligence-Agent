"""Opt-in remote LangSmith trace smoke test with generated synthetic data."""

import os
from datetime import datetime, timezone
from uuid import uuid4

import langsmith as ls
import pytest

from api.services.dataset_service import DatasetService
from api.services.query_service import QueryService
from core.guardrails.output_schema import OutputGuardrailResult
from core.guardrails.schema import GuardrailResult


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LANGSMITH_E2E") != "1" or not os.getenv("LANGSMITH_API_KEY"),
    reason="Set RUN_LANGSMITH_E2E=1 and LANGSMITH_API_KEY to verify remote trace delivery.",
)


class LocalInputGuardrail:
    def classify(self, question):
        return GuardrailResult(
            is_safe=True,
            is_prompt_injection=False,
            is_pii=False,
            is_irrelevant=False,
            reason="Synthetic trace test",
        )


class LocalOutputGuardrail:
    def classify(self, question, answer, evidence):
        return OutputGuardrailResult(
            is_safe=True,
            is_grounded=True,
            contains_unsupported_claims=False,
            contains_pii=False,
            reason="Synthetic trace test",
        )


def test_query_service_emits_a_remote_langsmith_trace():
    project = f"financial-intelligence-e2e-{uuid4()}"
    client = ls.Client(hide_inputs=True, hide_outputs=True)
    dataset_service = DatasetService()
    dataset_id, _ = dataset_service.load_dataset(
        "synthetic_trace.csv",
        b"Revenue,Profit\n100,12\n160,24\n",
    )
    service = QueryService(
        dataset_service,
        guardrail_classifier=LocalInputGuardrail(),
        output_guardrail_classifier=LocalOutputGuardrail(),
        reasoner=None,
    )
    started = datetime.now(timezone.utc)

    with ls.tracing_context(enabled=True, client=client, project_name=project):
        result = service.answer_query(
            dataset_id,
            "What was total revenue?",
            session_id="synthetic-trace-session",
        )
    assert result["answer"]
    client.flush(timeout=15)

    runs = list(client.list_runs(project_name=project, start_time=started, limit=50))
    names = {run.name for run in runs}
    assert "financial_query" in names
    assert "financial_agent_graph" in names
    assert {"guardrail", "query_understanding", "financial_analysis", "evidence_fusion", "financial_reasoning", "output_guardrail"} <= names
