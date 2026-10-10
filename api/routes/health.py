from fastapi import APIRouter
from api.services.container import query_service

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "financial-intelligence-agent"
    }


@router.get("/model/status")
def model_status():
    """Report configured reasoning model and whether a QLoRA adapter is active."""
    return query_service.model_status()
