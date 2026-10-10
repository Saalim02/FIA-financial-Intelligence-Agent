import base64
from api.services.fia_controller import FIAController
from api.services.voice_query_service import VoiceQueryService


class FakeVoiceService:
    def __init__(self, transcript):
        self.transcript = transcript
        self.synthesized = []

    def transcribe(self, _audio, _filename):
        return self.transcript

    def synthesize(self, answer):
        self.synthesized.append(answer)
        return b"mp3"


class FakeQueryService:
    def __init__(self):
        self.calls = []
        self.guardrail_classifier = None

    def answer_query(self, dataset_id, question, session_id=None):
        self.calls.append((dataset_id, question, session_id))
        return {"answer": "Backend financial answer", "intent": "financial", "route": "analytics"}


class FakeGuardrailClassifier:
    def __init__(self, irrelevant):
        self.irrelevant = irrelevant

    def classify(self, _question):
        return type("Classification", (), {
            "is_irrelevant": self.irrelevant,
            "is_safe": not self.irrelevant,
            "is_prompt_injection": False,
            "is_pii": False,
        })()


def make_service(transcript):
    voice = FakeVoiceService(transcript)
    query = FakeQueryService()
    service = VoiceQueryService(dataset_service=None, query_service=query, voice_service=voice)
    return service, voice, query


def test_identity_is_deterministic_and_does_not_require_dataset():
    service, voice, query = make_service("What is meant by FIA?")

    result = service.answer_voice_query(None, b"audio", "question.webm")

    assert result["answer"] == "FIA refers to Financial Intelligence Agent. How can I help you?"
    assert result["intent"] == "fia_identity"
    assert query.calls == []
    assert base64.b64decode(result["audio_base64"]) == b"mp3"
    assert voice.synthesized == [result["answer"]]


def test_capability_question_does_not_require_dataset():
    service, _, query = make_service("What is RAG?")

    result = service.answer_voice_query(None, b"audio", "question.webm")

    assert result["intent"] == "fia_capability"
    assert "retrieval-augmented generation" in result["answer"]
    assert query.calls == []


def test_financial_question_requires_dataset_and_is_not_sent_to_query_service():
    service, _, query = make_service("What were total sales?")

    result = service.answer_voice_query(None, b"audio", "question.webm")

    assert result["intent"] == "dataset_required"
    assert result["answer"] == "Please upload or select a financial dataset first."
    assert query.calls == []


def test_out_of_scope_question_uses_existing_backend_classifier_without_dataset():
    service, _, query = make_service("What is the capital of France?")
    query.guardrail_classifier = FakeGuardrailClassifier(irrelevant=True)

    result = service.answer_voice_query(None, b"audio", "question.webm")

    assert result["intent"] == "out_of_scope"
    assert "outside my scope" in result["answer"]
    assert query.calls == []


def test_financial_question_uses_existing_query_service_and_session():
    service, _, query = make_service("What were total sales?")

    result = service.answer_voice_query("dataset-1", b"audio", "question.webm", "session-1")

    assert result["answer"] == "Backend financial answer"
    assert query.calls == [("dataset-1", "What were total sales?", "session-1")]


def test_controller_recognizes_requested_identity_variation():
    assert FIAController().handle_identity("What is meant by FIA?")
