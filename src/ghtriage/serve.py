"""FastAPI service: /health, /model-info, /predict, /explain. Inputs are validated before scoring."""

from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ghtriage.registry import NoModelError, load_current, load_reference

MAX_BATCH = 1000


class PredictRequest(BaseModel):
    records: list[dict[str, float]] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    probability: float
    label: int


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class ExplainRequest(BaseModel):
    record: dict[str, float]


def _validate_record(record: dict[str, float], features: list[str], index: int = 0) -> None:
    missing = [f for f in features if f not in record]
    if missing:
        raise HTTPException(status_code=422, detail=f"Record {index} is missing features: {missing}")
    bad = [f for f in features if not math.isfinite(record[f])]
    if bad:
        raise HTTPException(status_code=422, detail=f"Record {index} has non-finite values: {bad}")


def create_app(registry_dir: str | None = None) -> FastAPI:
    registry = registry_dir or os.environ.get("GHTRIAGE_REGISTRY_DIR", "models")
    state: dict[str, Any] = {"model": None, "card": None, "version": None, "error": None, "explainer": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["model"], state["card"], state["version"] = load_current(registry)
        except (NoModelError, FileNotFoundError) as exc:
            state["error"] = str(exc)
        yield

    app = FastAPI(title="ghtriage model service", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail={"status": "no_model", "error": state["error"]})
        return {"status": "ok", "model_version": state["version"]}

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        if state["card"] is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        return state["card"]

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        features: list[str] = state["card"]["features"]
        for i, record in enumerate(req.records):
            _validate_record(record, features, i)
        frame = pd.DataFrame([{f: r[f] for f in features} for r in req.records])
        probs = state["model"].predict_proba(frame)[:, 1]
        threshold = state["card"].get("threshold", 0.5)
        return PredictResponse(
            model_version=state["version"],
            predictions=[Prediction(probability=float(p), label=int(p >= threshold)) for p in probs],
        )

    @app.post("/explain")
    def explain(req: ExplainRequest) -> dict[str, Any]:
        """Per-feature SHAP contributions for one record. Needs the optional 'explain' extra."""
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="No model loaded.")
        features: list[str] = state["card"]["features"]
        _validate_record(req.record, features)
        from ghtriage import explain as ex

        try:
            if state["explainer"] is None:
                state["explainer"] = ex.make_explainer(state["model"], load_reference(registry), features)
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        result = ex.explain_record(state["explainer"], req.record, features)
        frame = pd.DataFrame([{f: req.record[f] for f in features}])
        probability = float(state["model"].predict_proba(frame)[0, 1])
        return {"model_version": state["version"], "probability": probability, **result}

    return app


app = create_app()
