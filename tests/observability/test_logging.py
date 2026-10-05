import json
import logging

from core.observability.logging import (
    JSONFormatter,
    get_logger,
    log_event,
)


def test_json_formatter_outputs_valid_json():
    logger = logging.getLogger("test_logger")

    record = logger.makeRecord(
        "test_logger",
        logging.INFO,
        "",
        0,
        "test message",
        (),
        None,
    )

    record.event = "test_event"
    record.data = {
        "status": "ok",
    }

    output = JSONFormatter().format(record)
    payload = json.loads(output)

    assert payload["message"] == "test message"
    assert payload["event"] == "test_event"
    assert payload["data"]["status"] == "ok"
    assert "timestamp" in payload
    assert "level" in payload
    assert "request_id" in payload


def test_sensitive_values_are_redacted():
    logger = get_logger("test_security")

    class CaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.record = None

        def emit(self, record):
            self.record = record

    handler = CaptureHandler()
    logger.addHandler(handler)

    log_event(
        logger,
        logging.INFO,
        "security_test",
        api_key="super-secret-key",
        password="secret-password",
        authorization="Bearer secret-token",
        normal_value="visible",
    )

    output = JSONFormatter().format(handler.record)
    payload = json.loads(output)

    assert payload["data"]["api_key"] == "[REDACTED]"
    assert payload["data"]["password"] == "[REDACTED]"
    assert payload["data"]["authorization"] == "[REDACTED]"
    assert payload["data"]["normal_value"] == "visible"

    logger.removeHandler(handler)
