import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(the\s+)?system\s+prompt",
    r"print\s+(the\s+)?system\s+prompt",
    r"bypass\s+(the\s+)?(safety|security)\s+(rules|filters)",
]


def detect_prompt_injection(query: str) -> dict:
    """
    Detect common prompt-injection patterns.

    This is a deterministic first layer.
    It should not be treated as a complete security solution.
    """

    if not isinstance(query, str):
        return {
            "detected": False,
            "reason": None,
        }

    normalized_query = query.strip().lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_query):
            return {
                "detected": True,
                "reason": "Potential prompt injection detected.",
            }

    return {
        "detected": False,
        "reason": None,
    }
