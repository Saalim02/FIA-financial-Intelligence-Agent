import base64

from api.services.dataset_service import DatasetService
from api.services.query_service import QueryService
from api.services.voice_service import VoiceService


class VoiceQueryService:
    def __init__(
        self,
        dataset_service: DatasetService,
        query_service: QueryService,
        voice_service: VoiceService,
    ):
        self.dataset_service = dataset_service
        self.query_service = query_service
        self.voice_service = voice_service

    def answer_voice_query(
        self,
        dataset_id: str,
        audio_bytes: bytes,
        filename: str,
    ) -> dict:
        transcript = self.voice_service.transcribe(
            audio_bytes,
            filename,
        )

        result = self.query_service.answer_query(
            dataset_id,
            transcript,
        )

        answer = result.get("answer") or ""

        audio_bytes = self.voice_service.synthesize(answer)

        return {
            "dataset_id": dataset_id,
            "transcript": transcript,
            "answer": answer,
            "intent": result.get("intent"),
            "route": result.get("route"),
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_format": "mp3",
        }
