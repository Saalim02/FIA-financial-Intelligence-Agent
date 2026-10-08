from types import SimpleNamespace

from core.agent.llm_interface import LLMInterface
from api.services.voice_service import VoiceService


class FakeTranscriptions:
    def create(self, **kwargs):
        return SimpleNamespace(
            text="What was the revenue growth?"
        )


class FakeAudioResponse:
    def read(self):
        return b"fake-mp3-audio"


class FakeSpeech:
    def create(self, **kwargs):
        return FakeAudioResponse()


class FakeAudio:
    def __init__(self):
        self.transcriptions = FakeTranscriptions()
        self.speech = FakeSpeech()


class FakeOpenAI:
    def __init__(self):
        self.audio = FakeAudio()


def test_transcribe():
    service = VoiceService(client=FakeOpenAI())

    result = service.transcribe(
        b"fake-audio",
        "test.wav",
    )

    assert result == "What was the revenue growth?"


def test_synthesize():
    service = VoiceService(client=FakeOpenAI())

    result = service.synthesize(
        "What was the revenue growth?"
    )

    assert result == b"fake-mp3-audio"


def test_transcribe_rejects_empty_audio():
    service = VoiceService(client=FakeOpenAI())

    try:
        service.transcribe(b"", "test.wav")
        assert False
    except ValueError as exc:
        assert str(exc) == "Audio file is empty."


def test_synthesize_rejects_empty_text():
    service = VoiceService(client=FakeOpenAI())

    try:
        service.synthesize("")
        assert False
    except ValueError as exc:
        assert str(exc) == "Text for speech synthesis cannot be empty."

