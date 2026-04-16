"""POST /predict — helmet detection inference endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, UploadFile
from fastapi import status as http_status
from slowapi import Limiter  # type: ignore[import-untyped]
from slowapi.util import get_remote_address  # type: ignore[import-untyped]
from starlette.requests import Request

from api.schemas import PredictionResponse
from core.config import settings
from core.dependencies import get_inference_service, validate_upload_size
from core.logging import get_logger, log_extra
from services.ml_service import InferenceService

logger = get_logger(__name__)

limiter: Limiter = Limiter(key_func=get_remote_address)
router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Helmet detection inference",
    description=(
        "Accepts a single image file and returns all detected helmets with "
        "bounding boxes, confidence scores, and class labels. "
        f"Rate-limited to **{settings.rate_limit_predict}** per client IP."
    ),
    responses={
        422: {"description": "Unprocessable image file."},
        429: {"description": "Rate limit exceeded."},
        500: {"description": "Internal ML processing error."},
    },
)
@limiter.limit(settings.rate_limit_predict)
async def predict(
    # NOTE: `request` MUST be the first positional parameter — slowapi inspects
    # the function signature by name to locate the Starlette Request object.
    request: Request,
    file: UploadFile = Depends(validate_upload_size),
    svc: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Run helmet detection on the uploaded image."""
    t0: float = time.perf_counter()
    image_bytes: bytes = await file.read()
    file_size: int = len(image_bytes)

    logger.info(
        "Prediction request received.",
        extra=log_extra(
            file=file.filename,
            content_type=file.content_type,
            file_size_bytes=file_size,
        ),
    )

    result = await svc.predict(image_bytes)
    latency_ms: float = (time.perf_counter() - t0) * 1000

    logger.info(
        "Prediction response dispatched.",
        extra=log_extra(
            file=file.filename,
            file_size_bytes=file_size,
            latency_ms=round(latency_ms, 2),
            num_detections=result["num_detections"],
        ),
    )

    return PredictionResponse(
        num_detections=result["num_detections"],
        detections=result["detections"],
        latency_ms=round(latency_ms, 2),
    )
