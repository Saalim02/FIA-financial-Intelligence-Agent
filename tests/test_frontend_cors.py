import re

from api.cors import LOCAL_DEV_ORIGIN_REGEX


def test_vite_fallback_port_is_allowed_for_local_api_requests():
    assert re.fullmatch(LOCAL_DEV_ORIGIN_REGEX, "http://localhost:5174")
    assert re.fullmatch(LOCAL_DEV_ORIGIN_REGEX, "http://127.0.0.1:4317")
    assert not re.fullmatch(LOCAL_DEV_ORIGIN_REGEX, "https://example.com")
