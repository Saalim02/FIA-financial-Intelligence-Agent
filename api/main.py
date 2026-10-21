import time
import logging
import os
from uuid import uuid4

from core.observability.langsmith_config import configure_langsmith

configure_langsmith()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.cors import LOCAL_DEV_ORIGIN_REGEX
from api.routes.health import router as health_router
from api.routes.dataset import router as dataset_router
from api.routes.query import router as query_router
from api.routes.voice import router as voice_router

from core.observability.logging import (
    get_logger,
    log_event,
    set_request_id,
)


logger = get_logger(__name__)


app = FastAPI(
    title="Financial Intelligence Agent",
    description="Production-grade GenAI financial analysis API",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://3.6.100.42:5173",
    ).split(",")
    if origin.strip()
]
production_frontend_origin = "http://3.6.100.42:5173"
if production_frontend_origin not in allowed_origins:
    allowed_origins.append(production_frontend_origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Vite automatically selects the next port if 5173 is occupied.
    # Permit local development origins on any port while keeping the
    # configurable production allowlist explicit.
    allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid4()),
    )

    set_request_id(request_id)

    start = time.perf_counter()

    log_event(
        logger,
        20,
        "request_started",
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(request)

        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        response.headers["X-Request-ID"] = request_id

        log_event(
            logger,
            20,
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        log_event(
            logger,
            logging.ERROR,
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )

        raise


app.include_router(health_router)
app.include_router(dataset_router)
app.include_router(query_router)
app.include_router(voice_router)
