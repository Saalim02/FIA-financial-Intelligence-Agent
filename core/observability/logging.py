import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str):
    return _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


def _safe_value(value: Any) -> Any:
    sensitive_keys = {
        "api_key",
        "apikey",
        "authorization",
        "token",
        "access_token",
        "secret",
        "password",
    }

    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in sensitive_keys
            else _safe_value(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]

    return value


class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        if hasattr(record, "event"):
            payload["event"] = record.event

        if hasattr(record, "data"):
            payload["data"] = _safe_value(record.data)

        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__
            }

        return json.dumps(payload, default=str)


def configure_logging():
    root_logger = logging.getLogger()

    if getattr(root_logger, "_observability_configured", False):
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()

    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger.addHandler(handler)

    root_logger._observability_configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **data,
):
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "data": data,
        },
    )
