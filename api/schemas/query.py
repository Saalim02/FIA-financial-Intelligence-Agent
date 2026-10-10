from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=64)
    document_path: str | None = Field(default=None)

class QueryResult(BaseModel):
    answer: str | None = None
    result: Any = None
    evidence: list[dict] = Field(default_factory=list)
    intent: str | None = None
    route: str | None = None
    visualization: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    dataset_id: str
    question: str
    result: QueryResult
