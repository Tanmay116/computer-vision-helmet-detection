# 🪖 Helmet Detection API

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://docs.ultralytics.com/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade REST API** for real-time PPE helmet detection powered by a fine-tuned **YOLOv8m ONNX** model. Built to MLOps standards with structured GCP logging, per-IP rate limiting, tenacity-backed inference retries, and strict Pydantic V2 schemas throughout.

---

## ✨ Features

| Capability | Implementation |
|---|---|
| Object Detection | YOLOv8m via `ultralytics` (ONNX runtime) |
| Zero cold-start | Model loaded once at startup via FastAPI `lifespan` |
| Rate Limiting | `slowapi` — 60 requests/min per IP on `/predict` |
| Structured Logging | Google Cloud Logging (stdlib JSON fallback) |
| Resilient Inference | `tenacity` exponential back-off retries |
| Strict Validation | Pydantic V2 schemas for all endpoints |
| Static Typing | Full `mypy`/`ruff` compatible type annotations |
| Code Quality | Ruff linting + formatting (`pyproject.toml`) |
| Testing | `pytest-asyncio` + `httpx` ASGI client (no GPU needed) |

---

## 🗂️ Project Structure

```
cv_api/
├── main.py                  # App factory, lifespan, middleware, exception handlers
├── api/
│   ├── schemas.py           # Pydantic V2 request/response models
│   └── routes/
│       ├── health.py        # GET /health
│       └── predict.py       # POST /predict  (rate-limited)
├── core/
│   ├── config.py            # Pydantic BaseSettings — env-driven config
│   ├── logging.py           # GCP Cloud Logging / JSON stdout setup
│   ├── exceptions.py        # Custom exception types + FastAPI handlers
│   └── dependencies.py      # DI helpers: InferenceService holder, upload guard
├── services/
│   └── ml_service.py        # InferenceService: decode → predict → format
├── tests/
│   └── test_api.py          # Integration tests (mocked InferenceService)
├── pyproject.toml           # Ruff + pytest configuration
├── requirements.txt
└── .env.example
```

---

## ⚡ Quickstart

### 1. Prerequisites

- Python **3.11+**
- Your exported model file: `best.onnx` (place in the project root, or update `MODEL_PATH` in `.env`)

### 2. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set MODEL_PATH
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `best.onnx` | Path to the ONNX model file |
| `MODEL_DEVICE` | `cpu` | `cpu` \| `cuda:0` |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence |
| `RATE_LIMIT_PREDICT` | `60/minute` | slowapi limit string |
| `GCP_PROJECT_ID` | *(empty)* | Set to enable Cloud Logging |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | 10 MB upload cap |

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

---

## 🔌 API Reference

### `GET /health`

Returns service liveness and whether the model is loaded.

**Response `200 OK`**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true
}
```

---

### `POST /predict`

Accepts a single image upload and returns detected helmets.

**Rate limit**: 60 requests/minute per client IP.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | JPEG, PNG, BMP, or WEBP image |

**Response `200 OK`**
```json
{
  "num_detections": 2,
  "detections": [
    {
      "bbox": [120.5, 80.0, 300.25, 250.75],
      "confidence": 0.9312,
      "class_id": 0,
      "class_name": "helmet"
    }
  ],
  "latency_ms": 34.21
}
```

**Error responses**

| Status | `error` code | Trigger |
|---|---|---|
| `413` | `file_too_large` | Upload exceeds `MAX_UPLOAD_SIZE_BYTES` |
| `422` | `invalid_image` | Bytes cannot be decoded as an image |
| `429` | `rate_limit_exceeded` | Too many requests from this IP |
| `500` | `ml_processing_error` | Inference failed after all retries |

**Example — curl**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/construction_site.jpg"
```

**Example — Python `requests`**
```python
import requests

with open("construction_site.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("construction_site.jpg", f, "image/jpeg")},
    )
print(response.json())
```

---

## 🏗️ Architecture Deep-Dive

### Lifespan Model Loading

The YOLO model is loaded **once** at server boot via FastAPI's `lifespan` context manager. This eliminates cold-start latency on the first request and ensures clean teardown on shutdown.

```
Server Start
    └─ lifespan.__aenter__
        └─ InferenceService.load()
            ├─ YOLO("best.onnx")   ← loads ONNX into memory
            └─ warm-up prediction  ← initialises CUDA/ONNX graphs
```

### Decoupled Inference Service

Route handlers **never** contain ML logic. The `InferenceService` class in `services/ml_service.py` owns the full pipeline:

```
POST /predict
  └─ validate_upload_size (DI)     ← FileSizeExceededError if too large
  └─ get_inference_service (DI)    ← returns the warm singleton
  └─ predict route handler
      └─ InferenceService.predict()
          ├─ _decode_image()       ← cv2.imdecode → InvalidImageError
          └─ _run_inference()      ← @retry(tenacity) → MLProcessingError
              └─ _format_results() ← bbox / confidence / class_name dicts
```

### Retry Strategy

`tenacity` wraps `_run_inference` with exponential back-off:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4.0),
    after=after_log(logger, WARNING),
)
```

This handles transient CUDA OOM spikes and I/O lock contention without returning errors to the client.

### Logging

Every log call uses the `extra={}` parameter to attach structured context:

```python
logger.info("Prediction response dispatched.", extra={
    "filename": "site_a.jpg",
    "file_size_bytes": 204800,
    "latency_ms": 34.21,
    "num_detections": 2,
})

logger.error("Inference failed.", exc_info=True, extra={
    "image_shape": (1080, 1920, 3),
})
```

When `GCP_PROJECT_ID` is set, these log records are shipped to **Google Cloud Logging** as structured JSON entries and are queryable via Log Explorer.

---

## 🧪 Running Tests

Tests use a **mocked InferenceService** — no real model or GPU required.

```bash
pytest -v tests/
```

The test suite covers:
- Health endpoint schema validation
- Successful prediction response schema
- Invalid image → `422` response
- Missing file → `422` response

---

## 🐳 Docker (optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t helmet-detection-api .
docker run -p 8000:8000 --env-file .env helmet-detection-api
```
