from fastapi import APIRouter, HTTPException

from api.schemas.query import QueryRequest, QueryResponse
from api.services.container import dataset_service
from api.services.query_service import QueryService


router = APIRouter()

query_service = QueryService(dataset_service)


@router.post("/query", response_model=QueryResponse)
def query_dataset(request: QueryRequest):
    try:
        result = query_service.answer_query(
            request.dataset_id,
            request.question,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {exc}",
        )

    return QueryResponse(
        dataset_id=request.dataset_id,
        question=request.question,
        result=result,
    )