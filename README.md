# Lab1-MLOps_Homework

Production-ready sentiment analysis API for single-text inference.

## 1. Project Scope

The application exposes one endpoint:

- `POST /predict`

Input JSON:

```json
{ "text": "MLOps lab - homework - making prod app for sentiment analysis - very good" }
```

Output JSON:

```json
{ "prediction": "positive" }
```

Predictions are mapped from model classes:

- `0 -> negative`
- `1 -> neutral`
- `2 -> positive`

## 2. Tech Stack

- Python 3.12
- uv (dependency and environment management)
- FastAPI + Uvicorn
- sentence-transformers + torch
- scikit-learn + joblib
- pytest
- pre-commit (ruff + mypy)
- Docker + Docker Compose

## 3. Setup (uv)

Install dependencies and sync environment:

```bash
uv sync
```

Run local API:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. API Test (curl)

```bash
curl -X POST \
	'http://localhost:8000/predict' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{"text": "What a great MLOps lecture"}'
```

## 5. Testing

Run tests:

```bash
PYTHONPATH=. uv run --extra test pytest -q
```

Run quality checks:

```bash
uv run pre-commit run --all-files
```

## 6. Docker

Build and run with Docker Compose:

```bash
docker compose up --build -d
```

Check service status:

```bash
docker compose ps
```

Test endpoint from host:

```bash
curl -X POST \
	'http://localhost:8000/predict' \
	-H 'accept: application/json' \
	-H 'Content-Type: application/json' \
	-d '{"text": "What a great MLOps lecture, I am very satisfied"}'
```

Stop services:

```bash
docker compose down
```

## 7. Model Files

Expected model artifacts in `models/`:

- `models/sentence_transformer.model/`
- `models/classifier.joblib`

If model files are missing, app startup will fail with a clear error.