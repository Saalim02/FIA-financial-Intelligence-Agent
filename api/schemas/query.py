from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1)


class FinancialResult(BaseModel):
    total_revenue: float | None = None
    total_profit: float | None = None
    profit_margin: float | None = None


class QueryResponse(BaseModel):
    dataset_id: str
    question: str
    result: FinancialResult