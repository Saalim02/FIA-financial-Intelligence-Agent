from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1)


class QueryResult(BaseModel):
    answer: str | None = None
    result: Any = None
    evidence: list[dict] = Field(default_factory=list)
    intent: str | None = None
    route: str | None = None


class QueryResponse(BaseModel):
    dataset_id: str
    question: str
    result: QueryResult
