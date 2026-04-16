"""Decoupled ML inference service for helmet detection."""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from tenacity import (
    after_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from ultralytics import YOLO  # type: ignore[import-untyped]

from core.config import settings
from core.exceptions import InvalidImageError, MLProcessingError
from core.logging import get_logger, log_extra

logger = get_logger(__name__)


class InferenceService:
    """
    Encapsulates all ML logic for helmet detection inference.

    Responsibilities:
    - Load and hold the YOLOv8 ONNX model.
    - Decode raw image bytes via OpenCV/NumPy.
    - Run prediction with tenacity-backed retries.
    - Format raw YOLO results into structured dicts.

    This class is intentionally framework-agnostic — it has no FastAPI imports.
    """

    def __init__(self) -> None:
        self._model: YOLO | None = None

    # Lifecycle

    def load(self) -> None:
        """
        Load the ONNX model into memory.

        Called once during the application lifespan startup event so that the
        first HTTP request does not incur a cold-start penalty.
        """
        logger.info(
            "Loading YOLO model.",
            extra=log_extra(model_path=settings.model_path, device=settings.model_device),
        )
        self._model = YOLO(settings.model_path, task="detect")
        # Warm-up: run a blank frame through the model to initialise CUDA/ONNX graphs.
        blank: np.ndarray = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            blank,
            device=settings.model_device,
            conf=settings.model_confidence_threshold,
            iou=settings.model_iou_threshold,
            verbose=False,
        )
        logger.info("YOLO model loaded and warm-up complete.")

    def unload(self) -> None:
        """Release model reference so the GC can reclaim GPU/CPU memory."""
        self._model = None
        logger.info("YOLO model unloaded.")

    # Public API

    async def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Decode *image_bytes* and run helmet detection.

        Args:
            image_bytes: Raw bytes of the uploaded image file.

        Returns:
            A dict matching the ``PredictionResponse`` schema.

        Raises:
            InvalidImageError: If the bytes cannot be decoded as an image.
            MLProcessingError: If inference fails after all retries.
        """
        image: np.ndarray = self._decode_image(image_bytes)
        detections: list[dict[str, Any]] = self._run_inference(image)
        return {
            "num_detections": len(detections),
            "detections": detections,
        }

    # Private Helpers

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """
        Decode raw bytes into an OpenCV BGR numpy array.

        Args:
            image_bytes: Raw image bytes from the client upload.

        Raises:
            InvalidImageError: When OpenCV cannot decode the buffer.
        """
        np_arr: np.ndarray = np.frombuffer(image_bytes, dtype=np.uint8)
        image: np.ndarray | None = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError(
                "Uploaded file could not be decoded as a valid image. "
                "Supported formats: JPEG, PNG, BMP, WEBP."
            )
        return image

    @retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(
            min=settings.retry_wait_min_seconds,
            max=settings.retry_wait_max_seconds,
        ),
        after=after_log(logger, logging.WARNING),
        reraise=False,
    )
    def _run_inference(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        Execute YOLO prediction with tenacity exponential back-off retries.

        Retries handle transient memory or I/O lock errors that may occur under
        high-concurrency workloads (e.g., simultaneous CUDA memory access).

        Args:
            image: Decoded BGR numpy array.

        Raises:
            MLProcessingError: Propagated after all retries are exhausted.
        """
        if self._model is None:
            raise MLProcessingError("Model is not loaded. Call InferenceService.load() first.")

        t0: float = time.perf_counter()
        try:
            results = self._model.predict(
                image,
                device=settings.model_device,
                conf=settings.model_confidence_threshold,
                iou=settings.model_iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error(
                "YOLO inference call failed.",
                exc_info=True,
                extra=log_extra(image_shape=image.shape),
            )
            raise MLProcessingError(f"Inference failed: {exc}") from exc

        latency_ms: float = (time.perf_counter() - t0) * 1000
        logger.info(
            "Inference complete.",
            extra=log_extra(latency_ms=round(latency_ms, 2), num_results=len(results)),
        )

        return self._format_results(results)

    @staticmethod
    def _format_results(results: list[Any]) -> list[dict[str, Any]]:
        """
        Convert raw ultralytics :class:`Results` objects into serialisable dicts.

        Each detection dict contains:
        - ``bbox``: ``[x1, y1, x2, y2]`` in absolute pixel coordinates.
        - ``confidence``: Float rounded to 4 decimal places.
        - ``class_id``: Integer class index.
        - ``class_name``: String label from the model's names map.
        """
        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            names: dict[int, str] = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": int(box.cls[0]),
                        "class_name": names[int(box.cls[0])],
                    }
                )
        return detections
