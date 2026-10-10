from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.json_safety import json_safe
from api.schemas.dataset import DatasetUploadResponse
from api.services.container import dataset_service

router = APIRouter()
MAX_DATASET_SIZE = 50 * 1024 * 1024



@router.post(
    "/datasets/upload",
    response_model=DatasetUploadResponse
)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    try:
        contents = await file.read(MAX_DATASET_SIZE + 1)
        if len(contents) > MAX_DATASET_SIZE:
            raise ValueError("Dataset file exceeds the 50 MB limit.")

        dataset_id, df = dataset_service.load_dataset(
            file.filename,
            contents
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read dataset: {exc}"
        )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist()
    )


@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
):
    try:
        df = dataset_service.get_dataset(dataset_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    return json_safe({
        "dataset_id": dataset_id,
        "filename": dataset_service.get_filename(dataset_id),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_types": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "offset": offset,
        "limit": limit,
        "preview": df.iloc[offset:offset + limit].to_dict(orient="records"),
    })
