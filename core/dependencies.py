"""FastAPI dependency-injection helpers."""

from __future__ import annotations

from fastapi import UploadFile

from core.config import settings
from core.exceptions import FileSizeExceededError
from services.ml_service import InferenceService

# Global model holder (populated by lifespan)
# This avoids the overhead of reconstructing InferenceService on every request.
# The lifespan context manager in main.py is responsible for setting this.

_inference_service: InferenceService | None = None


def set_inference_service(service: InferenceService) -> None:
    """Store the singleton InferenceService created during startup."""
    global _inference_service  # noqa: PLW0603
    _inference_service = service


def get_inference_service() -> InferenceService:
    """
    FastAPI dependency that returns the pre-loaded :class:`InferenceService`.

    Raises:
        RuntimeError: If the service was not initialised during lifespan startup.
    """
    if _inference_service is None:
        raise RuntimeError(
            "InferenceService is not initialised. "
            "Ensure the lifespan context manager ran successfully."
        )
    return _inference_service


async def validate_upload_size(file: UploadFile) -> UploadFile:
    """
    FastAPI dependency that reads and validates the upload size.

    The file content is read into memory once and the position is reset so
    downstream handlers can read the bytes again.

    Raises:
        FileSizeExceededError: If the file exceeds ``settings.max_upload_size_bytes``.
    """
    content: bytes = await file.read()
    size: int = len(content)
    if size > settings.max_upload_size_bytes:
        raise FileSizeExceededError(max_bytes=settings.max_upload_size_bytes)
    # Reset the cursor so the route handler can read the bytes normally.
    await file.seek(0)
    return file
