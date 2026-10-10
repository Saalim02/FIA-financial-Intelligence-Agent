import pandas as pd

from api.services.query_service import QueryService
from core.agent.session_store import InMemorySessionStore


class FakeDatasetService:
    def get_dataset(self, dataset_id):
        return pd.DataFrame({"Revenue": [10, 20]})


class FakeGraph:
    def __init__(self):
        self.histories = []

    def invoke(self, state, config=None):
        self.histories.append(state["conversation_history"])
        return {
            "guardrail_safe": True,
            "output_guardrail_safe": True,
            "answer": f"answer {len(self.histories)}",
            "result": [],
            "evidence": [],
            "intent": "financial_analysis",
            "route": "financial",
        }


def make_service():
    service = QueryService.__new__(QueryService)
    service.dataset_service = FakeDatasetService()
    service.graph = FakeGraph()
    service.guardrail_classifier = object()
    service.output_guardrail_classifier = object()
    service.reasoner = object()
    service.session_store = InMemorySessionStore()
    return service


def test_session_history_is_passed_into_later_graph_turns():
    service = make_service()

    service.answer_query("dataset-1", "What was revenue?", session_id="session-1")
    service.answer_query("dataset-1", "And profit?", session_id="session-1")

    assert service.graph.histories[0] == []
    assert service.graph.histories[1] == [
        {"question": "What was revenue?", "answer": "answer 1"}
    ]


def test_session_context_is_isolated_when_dataset_changes():
    service = make_service()

    service.answer_query("dataset-1", "What was revenue?", session_id="session-1")
    service.answer_query("dataset-2", "What was profit?", session_id="session-1")

    assert service.graph.histories[1] == []


def test_session_history_is_bounded_to_ten_turns():
    service = make_service()

    for index in range(12):
        service.answer_query("dataset-1", f"Question {index}", session_id="session-1")

    history = service.session_store.history_for("session-1", "dataset-1")
    assert len(history) == 10
    assert history[0]["question"] == "Question 2"
