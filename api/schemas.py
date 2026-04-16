"""Pydantic V2 request/response schemas for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., examples=["ok"], description="Service liveness status.")
    version: str = Field(..., examples=["1.0.0"], description="Deployed API version.")
    model_loaded: bool = Field(..., description="Whether the YOLO model is currently in memory.")


class BoundingBox(BaseModel):
    """Detected object with bounding box, confidence, and class information."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        examples=[[120.5, 80.0, 300.25, 250.75]],
        description="[x1, y1, x2, y2] absolute pixel coordinates.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        examples=[0.9312],
        description="Detection confidence score.",
    )
    class_id: int = Field(..., ge=0, examples=[0], description="Numeric class index.")
    class_name: str = Field(..., examples=["helmet"], description="Human-readable class label.")


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    num_detections: int = Field(..., ge=0, description="Total number of objects detected.")
    detections: list[BoundingBox] = Field(default_factory=list, description="List of detected objects.")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end request latency in milliseconds.")


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by all exception handlers."""

    error: str = Field(..., examples=["invalid_image"], description="Machine-readable error code.")
    detail: str = Field(..., examples=["Cannot decode uploaded file."], description="Human-readable message.")
