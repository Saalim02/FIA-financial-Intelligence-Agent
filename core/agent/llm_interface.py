import logging
import time


from openai import OpenAI

from core.observability.logging import get_logger, log_event

logger = get_logger(__name__)

class LLMInterface:

    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt, user_prompt):
        start = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "llm_call_started",
            model=self.model,
            operation="generate",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            if not response.choices:
                log_event(
                    logger,
                    logging.WARNING,
                    "llm_call_empty",
                    model=self.model,
                    operation="generate",
                    latency_ms=latency_ms,
                )
                return ""

            message = response.choices[0].message

            if not message or not message.content:
                log_event(
                    logger,
                    logging.WARNING,
                    "llm_call_empty",
                    model=self.model,
                    operation="generate",
                    latency_ms=latency_ms,
                )
                return ""

            log_event(
                logger,
                logging.INFO,
                "llm_call_completed",
                model=self.model,
                operation="generate",
                latency_ms=latency_ms,
            )

            return message.content

        except Exception as exc:
            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.ERROR,
                "llm_call_failed",
                model=self.model,
                operation="generate",
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

            raise

    def generate_structured(self, system_prompt, user_prompt, response_model):
        start = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "llm_call_started",
            model=self.model,
            operation="generate_structured",
            response_model=response_model.__name__,
        )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format=response_model,
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            if not response.choices:
                raise RuntimeError("LLM returned no choices.")

            parsed = response.choices[0].message.parsed

            if parsed is None:
                raise RuntimeError("LLM returned no structured response.")

            log_event(
                logger,
                logging.INFO,
                "llm_call_completed",
                model=self.model,
                operation="generate_structured",
                response_model=response_model.__name__,
                latency_ms=latency_ms,
            )

            return parsed

        except Exception as exc:
            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.ERROR,
                "llm_call_failed",
                model=self.model,
                operation="generate_structured",
                response_model=response_model.__name__,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

            raise RuntimeError(
                f"Structured LLM call failed: {type(exc).__name__}"
            ) from exc