"""Optional real-provider smoke test using only generated synthetic data.

Run explicitly with RUN_PAID_E2E=1 after configuring OPENAI_API_KEY. This sends
the synthetic CSV and synthetic speech prompt to the configured OpenAI services.
"""

import os

import pytest

from api.services.dataset_service import DatasetService
from api.services.query_service import QueryService
from api.services.voice_query_service import VoiceQueryService
from api.services.voice_service import VoiceService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PAID_E2E") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set RUN_PAID_E2E=1 and OPENAI_API_KEY to call paid OpenAI services.",
)


def test_real_text_and_voice_round_trip_share_one_session():
    dataset_service = DatasetService()
    csv = b"Month,Revenue,Profit\n2025-01,100,12\n2025-02,140,19\n2025-03,190,31\n"
    dataset_id, _ = dataset_service.load_dataset("synthetic_paid_e2e.csv", csv)

    query_service = QueryService(dataset_service)
    voice_service = VoiceService()
    voice_query_service = VoiceQueryService(
        dataset_service=dataset_service,
        query_service=query_service,
        voice_service=voice_service,
    )
    session_id = "synthetic-paid-e2e"

    text_result = query_service.answer_query(
        dataset_id,
        "What was the total revenue?",
        session_id=session_id,
    )
    assert text_result["answer"]
    assert text_result["result"]

    speech_prompt = "What about profit?"
    audio = voice_service.synthesize(speech_prompt)
    voice_result = voice_query_service.answer_voice_query(
        dataset_id=dataset_id,
        audio_bytes=audio,
        filename="synthetic-question.mp3",
        session_id=session_id,
    )
    assert voice_result["transcript"]
    assert voice_result["answer"]
    assert voice_result["audio_base64"]
    assert voice_result["dataset_id"] == dataset_id
