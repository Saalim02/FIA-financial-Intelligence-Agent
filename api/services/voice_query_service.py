import base64

from api.services.fia_controller import FIAController


class VoiceQueryService:
    def __init__(self, dataset_service, query_service, voice_service):
        self.dataset_service = dataset_service
        self.query_service = query_service
        self.voice_service = voice_service
        self.fia = FIAController()

    def _synthesize_response(self, answer: str) -> bytes:
        return self.voice_service.synthesize(answer)

    def answer_voice_query(
        self,
        dataset_id,
        audio_bytes,
        filename,
        session_id=None,
    ):
        transcript = self.voice_service.transcribe(
            audio_bytes,
            filename,
        )

        # FIA identity and capability answers are deterministic and do not
        # need a dataset or a reasoning-model call.
        identity_answer = self.fia.handle_identity(transcript)
        capability_answer = self.fia.handle_capability(transcript)

        if identity_answer:
            answer = identity_answer
            result = {
                "intent": "fia_identity",
                "route": "fia_controller",
            }
        elif capability_answer:
            answer = capability_answer
            result = {
                "intent": "fia_capability",
                "route": "fia_controller",
            }
        elif not dataset_id:
            classifier = getattr(self.query_service, "guardrail_classifier", None)
            try:
                classification = classifier.classify(transcript) if classifier else None
            except Exception:
                # Without a classification result, fail closed on data access
                # while still giving the user a useful next step.
                classification = None
            if classification and (
                classification.is_irrelevant
                or not classification.is_safe
                or classification.is_prompt_injection
                or classification.is_pii
            ):
                answer = self.fia.out_of_scope_response()
                result = {"intent": "out_of_scope", "route": "fia_controller"}
            else:
                answer = self.fia.missing_dataset_response()
                result = {"intent": "dataset_required", "route": "fia_controller"}
        else:
            # All legitimate financial questions continue through
            # the existing production intelligence pipeline.
            result = self.query_service.answer_query(
                dataset_id,
                transcript,
                session_id=session_id,
            )

            answer = result.get("answer") or ""

        audio_bytes = self._synthesize_response(answer)

        return {
            "dataset_id": dataset_id,
            "transcript": transcript,
            "answer": answer,
            "intent": result.get("intent"),
            "route": result.get("route"),
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_format": "mp3",
            "visualization": result.get("visualization"),
        }
