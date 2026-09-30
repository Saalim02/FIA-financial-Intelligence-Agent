from pydantic import BaseModel, Field


class OutputGuardrailResult(BaseModel):
    is_safe: bool = Field(
        description="Whether the generated answer is safe to return."
    )

    is_grounded: bool = Field(
        description="Whether the answer is supported by the supplied evidence."
    )

    contains_unsupported_claims: bool = Field(
        description="Whether the answer contains claims not supported by the evidence."
    )

    contains_pii: bool = Field(
        description="Whether the answer exposes sensitive personally identifiable information."
    )

    reason: str = Field(
        description="Short explanation for the output classification."
    )
