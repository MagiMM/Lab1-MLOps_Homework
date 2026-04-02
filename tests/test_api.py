import pytest
from fastapi.testclient import TestClient

import main


def test_model_loads_on_startup() -> None:
    # model should be loaded from provided cloudpickle file without errors
    with TestClient(main.app):
        assert main.sentence_model is not None
        assert main.classifier is not None
        assert hasattr(main.classifier, "predict")


def test_predict_returns_valid_json_response() -> None:
    # output should be a valid JSON response
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"text": "What a great MLOps lecture"})

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"prediction"}
    assert payload["prediction"] in {"negative", "neutral", "positive"}


def test_inference_works_for_sample_strings() -> None:
    # inference should work for a few sample strings
    sample_texts = [
        "I love this course",
        "This is fine",
        "I hate waiting for broken builds",
    ]
    with TestClient(main.app) as client:
        for text in sample_texts:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            assert response.json()["prediction"] in {
                "negative",
                "neutral",
                "positive",
            }


@pytest.mark.parametrize("invalid_payload", [{"text": ""}, {"text": "   "}, {}])
def test_invalid_input_returns_validation_error_json(invalid_payload: dict) -> None:
    # input text should be a non-empty string
    # for an invalid input, validation should return a valid JSON with error explanation
    with TestClient(main.app) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
    payload = response.json()
    assert isinstance(payload, dict)
    assert "detail" in payload
    assert isinstance(payload["detail"], list)
