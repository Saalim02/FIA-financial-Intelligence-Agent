from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas.voice import VoiceResponse
from api.services.container import dataset_service, query_service, voice_service
from api.services.voice_query_service import VoiceQueryService

router = APIRouter()

MAX_AUDIO_SIZE = 10 * 1024 * 1024

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
}

voice_query_service = VoiceQueryService(
    dataset_service=dataset_service,
    query_service=query_service,
    voice_service=voice_service,
)


@router.post("/voice/query", response_model=VoiceResponse)
async def voice_query(
    dataset_id: str | None = Form(None),
    session_id: str | None = Form(None, max_length=64),
    audio: UploadFile = File(...),
):
    try:
        if audio.content_type and audio.content_type not in ALLOWED_AUDIO_TYPES:
            raise ValueError(
                f"Unsupported audio type: {audio.content_type}"
            )

        audio_bytes = await audio.read()

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise ValueError("Audio file exceeds the 10 MB limit.")

        result = voice_query_service.answer_voice_query(
            dataset_id=dataset_id,
            audio_bytes=audio_bytes,
            filename=audio.filename or "audio.wav",
            session_id=session_id,
        )

        return VoiceResponse(**result)

    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Voice query failed.",
        )
