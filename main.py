from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent / "models"
SENTENCE_MODEL_PATH = MODEL_DIR / "sentence_transformer.model"
CLASSIFIER_PATH = MODEL_DIR / "classifier.joblib"
CLASS_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

sentence_model: SentenceTransformer | None = None
classifier: Any = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global sentence_model, classifier

    if not SENTENCE_MODEL_PATH.exists() or not CLASSIFIER_PATH.exists():
        raise RuntimeError(
            "Model files were not found. Expected files in 'models/' directory."
        )

    sentence_model = SentenceTransformer(str(SENTENCE_MODEL_PATH))
    classifier = joblib.load(CLASSIFIER_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if sentence_model is None or classifier is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    embedding = sentence_model.encode([request.text])
    prediction_id = int(classifier.predict(embedding)[0])
    prediction = CLASS_LABELS.get(prediction_id)

    if prediction is None:
        raise HTTPException(status_code=500, detail="Unknown model output class.")

    return PredictResponse(prediction=prediction)
