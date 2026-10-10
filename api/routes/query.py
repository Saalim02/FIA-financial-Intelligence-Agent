from fastapi import APIRouter, HTTPException

from api.json_safety import json_safe
from api.schemas.query import QueryRequest, QueryResponse
from api.services.container import dataset_service, query_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_dataset(request: QueryRequest):
    try:
        result = query_service.answer_query(
            request.dataset_id,
            request.question,
            session_id=request.session_id,
            document_path=request.document_path,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

    return QueryResponse(
        dataset_id=request.dataset_id,
        question=request.question,
        result=json_safe(result),
    )
