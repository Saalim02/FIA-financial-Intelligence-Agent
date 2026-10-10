import os

import langsmith as ls
from dotenv import load_dotenv


def configure_langsmith() -> None:
    # Load project settings before creating the client: api.main invokes this
    # before importing services, and .env may contain tracing configuration.
    load_dotenv()
    enabled = os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        ls.configure(enabled=False)
        return

    client = ls.Client(
        timeout_ms=(30_000, 180_000),
        max_batch_size_bytes=1_000_000,
        hide_inputs=True,
        hide_outputs=True,
    )

    ls.configure(
        client=client,
        enabled=True,
        project_name=os.getenv("LANGSMITH_PROJECT", "financial-intelligence-agent"),
    )
