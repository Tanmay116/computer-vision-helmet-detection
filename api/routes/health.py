"""GET /health — liveness/readiness check."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import status as http_status

from api.schemas import HealthResponse
from core.config import settings
from core.dependencies import get_inference_service

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Service health check",
    description="Returns the current operational status of the service and whether the model is loaded.",
)
async def health_check() -> HealthResponse:
    """Return service liveness status and model availability."""
    try:
        svc = get_inference_service()
        model_loaded: bool = svc._model is not None  # noqa: SLF001
    except RuntimeError:
        model_loaded = False

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model_loaded=model_loaded,
    )
