# 🪖 Helmet Detection API

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://docs.ultralytics.com/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade REST API** for real-time PPE helmet detection powered by a fine-tuned **YOLOv8m** model. Built to MLOps standards with structured JSON logging, per-IP rate limiting, tenacity-backed inference retries, and strict Pydantic V2 schemas throughout.

---

## ✨ Features

| Capability | Implementation |
|---|---|
| Object Detection | YOLOv8m via `ultralytics` |
| Zero cold-start | Model loaded once at startup via FastAPI `lifespan` |
| Rate Limiting | `slowapi` — 60 requests/min per IP on `/predict` |
| Structured Logging | JSON Lines (stdout + `app.log`) |
| Resilient Inference | `tenacity` exponential back-off retries |
| Strict Validation | Pydantic V2 schemas for all endpoints |
| Static Typing | Full `mypy`/`ruff` compatible type annotations |
| Code Quality | Ruff linting + formatting (`pyproject.toml`) |
| Testing | `pytest` + `httpx` ASGI client (mocked inference) |

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
│   ├── logging.py           # Structured JSON logging factory
│   ├── exceptions.py        # Custom exception types + FastAPI handlers
│   └── dependencies.py      # DI: InferenceService holder, upload validator
├── services/
│   └── ml_service.py        # InferenceService: decode → predict → format
├── tests/
│   └── test_api.py          # Integration tests (mocked InferenceService)
├── pyproject.toml           # Ruff + pytest configuration
├── requirements.txt
├── best.pt                  # Fine-tuned YOLOv8m weights (~50 MB)
└── .env.example
```

---

## ⚡ Quickstart

### 1. Prerequisites

- Python **3.12.x**
- Your fine-tuned model file: `best.pt` (place in the project root)

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
# Edit .env if needed (defaults are production-ready)
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `best.pt` | Path to the YOLO weights file |
| `MODEL_DEVICE` | `cpu` | `cpu` | `cuda:0` |
| `MODEL_CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence |
| `RATE_LIMIT_PREDICT` | `60/minute` | slowapi limit string |
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

**Rate limit**: 60 requests/minute per client IP (configurable).

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | JPEG, PNG, BMP, or WEBP image (≤10 MB) |

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

---

## 🏗️ Architecture Deep-Dive

### Lifespan Model Loading

The YOLO model is loaded **once** at server boot via FastAPI's `lifespan` context manager. A warm-up forward pass is executed to initialise CUDA/CPU graphs.

```
Server Start
    └─ lifespan.__aenter__
        └─ InferenceService.load()
            ├─ YOLO("best.pt")      ← loads weights into memory
            └─ warm-up prediction   ← initialises compute graphs
```

### Decoupled Inference Service

Route handlers are decoupled from ML logic. The `InferenceService` owns the full pipeline:

```
POST /predict
  └─ validate_upload_size (DI)     ← FileSizeExceededError if > 10MB
  └─ get_inference_service (DI)    ← returns singleton service
  └─ predict route handler
      └─ InferenceService.predict()
          ├─ _decode_image()       ← cv2.imdecode → np.ndarray
          └─ _run_inference()      ← @retry(tenacity) → MLProcessingError
              └─ _format_results() ← round metrics & map class labels
```

### Retry Strategy

`tenacity` wraps inference with exponential back-off to handle transient resource contention:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4.0),
    after=after_log(logger, WARNING),
)
```

### Observability

Logs are emitted as JSON Lines for easy ingestion and querying:

```python
logger.info("Prediction complete.", extra={
    "latency_ms": 34.21,
    "num_detections": 2,
    "file_size": 204800,
})
```

When deployed to GCP, these structured logs are automatically indexed by **Cloud Logging**.

---

## 🧪 Testing

```bash
pytest -v tests/
```

The test suite uses a **mocked InferenceService**, allowing for rapid CI validation without requiring a GPU or large model files in the test environment.

---

## 🐳 Docker

```bash
docker build -t helmet-detection-api .
docker run -p 8000:8000 --env-file .env helmet-detection-api
```
