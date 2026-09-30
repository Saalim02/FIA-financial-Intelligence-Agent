MAX_QUERY_LENGTH = 2000


def validate_input(query: str) -> dict:
    """
    Validate user input before it enters the agent workflow.
    """

    if not isinstance(query, str):
        return {
            "valid": False,
            "reason": "Query must be a string.",
        }

    query = query.strip()

    if not query:
        return {
            "valid": False,
            "reason": "Query cannot be empty.",
        }

    if len(query) > MAX_QUERY_LENGTH:
        return {
            "valid": False,
            "reason": f"Query exceeds the maximum length of {MAX_QUERY_LENGTH} characters.",
        }

    return {
        "valid": True,
        "reason": None,
        "query": query,
    }
