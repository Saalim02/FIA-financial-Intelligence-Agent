from pydantic import BaseModel


class VoiceResponse(BaseModel):
    dataset_id: str
    transcript: str
    answer: str
    intent: str | None = None
    route: str | None = None
    audio_base64: str
    audio_format: str = "mp3"
