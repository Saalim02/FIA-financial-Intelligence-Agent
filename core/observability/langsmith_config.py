import langsmith as ls


def configure_langsmith() -> None:
    client = ls.Client(
        timeout_ms=(30_000, 180_000),
        max_batch_size_bytes=1_000_000,
        hide_inputs=True,
        hide_outputs=True,
    )

    ls.configure(
        client=client,
        enabled=True,
        project_name="financial-intelligence-agent",
    )
