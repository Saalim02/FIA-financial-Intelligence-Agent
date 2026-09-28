from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.dataset import router as dataset_router
from api.routes.query import router as query_router


app = FastAPI(
    title="Financial Intelligence Agent",
    description="Production-grade GenAI financial analysis API",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(query_router)