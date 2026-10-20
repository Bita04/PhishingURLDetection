"""Local FastAPI service for the explainable phishing URL classifier."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from explain import explain_url

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "phishing_model.joblib"

app = FastAPI(
    title="Explainable Phishing URL Detector",
    description="Local, lexical-only URL classification. The submitted site is never fetched.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)

    @field_validator("url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as error:
            raise ValueError("Provide a valid absolute HTTP(S) URL.") from error
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not hostname
            or any(character.isspace() for character in hostname)
        ):
            raise ValueError("URL must include an http:// or https:// scheme and hostname.")
        return value


class PredictionResponse(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1, description="Estimated phishing probability.")
    reasons: list[str]


@lru_cache(maxsize=1)
def load_predictor() -> tuple[dict, shap.LinearExplainer]:
    model_path = Path(os.environ.get("PHISHING_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model missing at {model_path}. Run `python backend/train.py` first."
        )
    bundle = joblib.load(model_path)
    estimator = bundle["models"][bundle["default_model"]]
    explainer = shap.LinearExplainer(estimator[-1], bundle["shap_background"])
    return bundle, explainer


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle, explainer = load_predictor()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    estimator = bundle["models"][bundle["default_model"]]
    result = explain_url(estimator, explainer, request.url)
    return PredictionResponse(**result)
