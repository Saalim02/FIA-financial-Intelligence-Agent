from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas.dataset import DatasetUploadResponse
from api.services.container import dataset_service

router = APIRouter()



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
        contents = await file.read()

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
def get_dataset(dataset_id: str):
    try:
        df = dataset_service.get_dataset(dataset_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )

    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records")
    }
