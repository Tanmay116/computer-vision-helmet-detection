"""Custom application exception types and FastAPI exception handlers."""

from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded  # type: ignore[import-untyped]

from core.logging import get_logger

logger = get_logger(__name__)

# Domain Exceptions


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as a valid image."""

    def __init__(self, detail: str = "Cannot decode uploaded file as a valid image.") -> None:
        self.detail = detail
        super().__init__(detail)


class MLProcessingError(RuntimeError):
    """Raised when the inference pipeline fails for a non-transient reason."""

    def __init__(self, detail: str = "Internal ML processing error.") -> None:
        self.detail = detail
        super().__init__(detail)


class FileSizeExceededError(ValueError):
    """Raised when the uploaded file exceeds the configured size limit."""

    def __init__(self, max_bytes: int) -> None:
        self.detail = f"Uploaded file exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB."
        super().__init__(self.detail)


# FastAPI Exception Handlers


async def invalid_image_handler(request: Request, exc: InvalidImageError) -> JSONResponse:
    logger.warning(
        "Invalid image uploaded.",
        extra={"path": str(request.url), "detail": exc.detail},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "invalid_image", "detail": exc.detail},
    )


async def ml_processing_handler(request: Request, exc: MLProcessingError) -> JSONResponse:
    logger.error(
        "ML processing error.",
        exc_info=True,
        extra={"path": str(request.url), "detail": exc.detail},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "ml_processing_error", "detail": exc.detail},
    )


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning(
        "Rate limit exceeded.",
        extra={"client_ip": request.client.host if request.client else "unknown", "path": str(request.url)},
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}. Please slow down.",
        },
    )


async def file_size_handler(request: Request, exc: FileSizeExceededError) -> JSONResponse:
    logger.warning(
        "File size exceeded.",
        extra={"path": str(request.url), "detail": exc.detail},
    )
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={"error": "file_too_large", "detail": exc.detail},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception.",
        exc_info=True,
        extra={"path": str(request.url), "method": request.method},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )
