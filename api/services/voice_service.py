import io
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class VoiceService:
    """Handles speech-to-text and text-to-speech operations."""

    def __init__(
        self,
        client: OpenAI | None = None,
        transcription_model: str = "gpt-4o-mini-transcribe",
        speech_model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
    ):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key and client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the Voice Service."
            )

        self.client = client or OpenAI(api_key=api_key)
        self.transcription_model = transcription_model
        self.speech_model = speech_model
        self.voice = voice

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Convert uploaded audio into text."""

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        response = self.client.audio.transcriptions.create(
            model=self.transcription_model,
            file=audio_file,
        )

        transcript = getattr(response, "text", None)

        if not transcript:
            raise RuntimeError("Speech-to-text returned an empty transcript.")

        return transcript.strip()

    def synthesize(self, text: str) -> bytes:
        """Convert text into MP3 audio."""

        if not text or not text.strip():
            raise ValueError("Text for speech synthesis cannot be empty.")

        response = self.client.audio.speech.create(
            model=self.speech_model,
            voice=self.voice,
            input=text.strip(),
            response_format="mp3",
        )

        return response.read()
