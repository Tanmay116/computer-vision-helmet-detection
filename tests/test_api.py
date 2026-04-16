"""
Integration tests for the Helmet Detection API.

Run with:
    pytest -v tests/test_api.py

These tests use FastAPI's ASGI test client (via httpx) so no real server is needed.
The InferenceService is mocked so no GPU/model file is required in CI.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_inference_service() -> MagicMock:
    """A mock InferenceService that returns a single helmet detection."""
    svc = MagicMock()
    svc._model = MagicMock()  # Marks the model as "loaded" for the health check.
    svc.predict = AsyncMock(
        return_value={
            "num_detections": 1,
            "detections": [
                {
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                    "confidence": 0.9312,
                    "class_id": 0,
                    "class_name": "helmet",
                }
            ],
        }
    )
    return svc


@pytest.fixture()
def client(mock_inference_service: MagicMock) -> TestClient:
    """Test client with the InferenceService dependency overridden."""
    from core.dependencies import get_inference_service

    app.dependency_overrides[get_inference_service] = lambda: mock_inference_service
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ── Minimal valid JPEG (1×1 pixel) for upload tests ───────────────────────────
TINY_JPEG: bytes = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),\x01\x02\x03"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
    b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\x28\xa2\x8a\xff\xd9"
)


# ── Health Tests ──────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        data: dict[str, Any] = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert isinstance(data["model_loaded"], bool)


# ── Predict Tests ─────────────────────────────────────────────────────────────


class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_image(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", TINY_JPEG, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        data: dict[str, Any] = client.post(
            "/predict",
            files={"file": ("test.jpg", TINY_JPEG, "image/jpeg")},
        ).json()
        assert "num_detections" in data
        assert "detections" in data
        assert "latency_ms" in data
        assert data["num_detections"] == 1
        det = data["detections"][0]
        assert len(det["bbox"]) == 4
        assert 0.0 <= det["confidence"] <= 1.0

    def test_predict_rejects_invalid_image(
        self, client: TestClient, mock_inference_service: MagicMock
    ) -> None:
        from core.exceptions import InvalidImageError

        mock_inference_service.predict = AsyncMock(side_effect=InvalidImageError())
        response = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_image"

    def test_predict_missing_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict")
        assert response.status_code == 422
