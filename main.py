"""
Helmet Detection API — Application Entry Point.

Startup / shutdown lifecycle:
  1. Load YOLOv8 ONNX model into memory.
  2. Register a singleton InferenceService in the DI layer.
  3. On shutdown, release model memory.

Middleware:
  - slowapi RateLimitMiddleware for per-IP throttling.
  - CORS (permissive by default; tighten in production).

Exception handlers are registered for all custom domain errors.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore[import-untyped]
from slowapi.errors import RateLimitExceeded  # type: ignore[import-untyped]
from slowapi.middleware import SlowAPIMiddleware  # type: ignore[import-untyped]
from slowapi.util import get_remote_address  # type: ignore[import-untyped]

from api.routes import health, predict
from core.config import settings
from core.dependencies import set_inference_service
from core.exceptions import (
    FileSizeExceededError,
    InvalidImageError,
    MLProcessingError,
    file_size_handler,
    invalid_image_handler,
    ml_processing_handler,
    rate_limit_handler,
    unhandled_exception_handler,
)
from core.logging import get_logger
from services.ml_service import InferenceService

startup_logger = get_logger("startup")

# Rate limiter (shared between middleware and route decorators)
limiter: Limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# Lifespan

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """
    Manage the application lifecycle.

    Startup:
        - Instantiate and load ``InferenceService`` (warm YOLO ONNX model).
        - Store the singleton via ``set_inference_service``.

    Shutdown:
        - Call ``InferenceService.unload()`` to release GPU/CPU memory.
    """
    startup_logger.info(
        "Application startup: loading inference service.",
        extra={"model_path": settings.model_path, "device": settings.model_device},
    )
    svc = InferenceService()
    try:
        svc.load()
    except Exception:
        startup_logger.error(
            "Failed to load YOLO model during startup.",
            exc_info=True,
            extra={"model_path": settings.model_path},
        )
        raise

    set_inference_service(svc)
    startup_logger.info("Inference service ready. Server is accepting requests.")

    yield  # Server is running

    startup_logger.info("Application shutdown: releasing model memory.")
    svc.unload()


# Application Factory

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade REST API for real-time PPE helmet detection "
            "powered by YOLOv8m (ONNX)."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Middleware
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict to specific origins in production.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(InvalidImageError, invalid_image_handler)  # type: ignore[arg-type]
    app.add_exception_handler(MLProcessingError, ml_processing_handler)  # type: ignore[arg-type]
    app.add_exception_handler(FileSizeExceededError, file_size_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)

    return app


app: FastAPI = create_app()
