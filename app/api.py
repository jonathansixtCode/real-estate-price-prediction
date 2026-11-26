"""
FastAPI Property Valuation Service — Ames Housing Price Estimator
Run from project root: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Dict
import os
import sys
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.deployment import predict_with_confidence
except ImportError:
    from deployment import predict_with_confidence


def resolve_bundle_path() -> str:
    env_path = os.getenv("MODEL_BUNDLE_PATH")
    candidates = [p for p in [env_path, "models/ames_housing_bundle.joblib", os.path.join(PROJECT_ROOT, "models/ames_housing_bundle.joblib"), os.path.join(HERE, "..", "models/ames_housing_bundle.joblib")] if p]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0] if candidates else "{bundle_rel_path}"


BUNDLE_PATH = resolve_bundle_path()

app = FastAPI(
    title="Ames Real Estate Valuation API",
    description="Predicts residential property sale prices with calibrated confidence intervals.",
    version="1.0.0",
)

model_bundle = None


@app.on_event("startup")
def load_model() -> None:
    global model_bundle
    path = resolve_bundle_path()
    if not os.path.exists(path):
        raise RuntimeError(f"Model bundle not found at {path}. Run Module 6 packaging first.")
    model_bundle = joblib.load(path)


class ValuationRequest(BaseModel):
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scaled feature values (unspecified features default to training medians).",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.50,
        le=0.99,
        description="Confidence level for prediction interval (e.g., 0.90 or 0.95).",
    )


class ConfidenceRange(BaseModel):
    lower_bound: float
    upper_bound: float


class ValuationResponse(BaseModel):
    estimated_price_usd: float
    confidence_level: float
    confidence_range_usd: ConfidenceRange
    log_prediction: float
    model_type: str


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "model_loaded": str(model_bundle is not None)}


@app.get("/features")
def list_features() -> Dict[str, object]:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    return {
        "n_features": len(model_bundle["feature_names"]),
        "feature_names": model_bundle["feature_names"],
        "feature_medians": model_bundle["feature_medians"],
    }


@app.post("/predict", response_model=ValuationResponse)
def predict_price(payload: ValuationRequest) -> ValuationResponse:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle not loaded.")
    result = predict_with_confidence(
        bundle=model_bundle,
        input_features=payload.features,
        confidence_level=payload.confidence_level,
    )
    return ValuationResponse(**result)
