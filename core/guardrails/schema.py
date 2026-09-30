from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    is_safe: bool = Field(
        description="Whether the user query is safe to process."
    )

    is_prompt_injection: bool = Field(
        description="Whether the query attempts to manipulate or override model instructions."
    )

    is_pii: bool = Field(
        description="Whether the query contains sensitive personally identifiable information."
    )

    is_irrelevant: bool = Field(
        description="Whether the query is unrelated to the financial intelligence system."
    )

    reason: str = Field(
        description="Short explanation for the classification."
    )
