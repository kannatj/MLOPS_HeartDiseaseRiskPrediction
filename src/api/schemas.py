"""Pydantic schemas for the Heart Disease API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Input schema mirroring the raw UCI columns (post-cleanup).

    Every field is optional because the trained pipeline imputes missing
    values, but sending nulls should be a deliberate choice by the client.
    """

    model_config = ConfigDict(extra="ignore")

    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    sex: Literal["Male", "Female"] = Field(..., description="Patient sex")
    dataset: Literal["Cleveland", "Hungary", "Switzerland", "VA Long Beach"] = Field(
        "Cleveland", description="Source hospital"
    )
    cp: Literal[
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic",
    ] = Field(..., description="Chest pain type")
    trestbps: float | None = Field(None, ge=0, description="Resting blood pressure (mm Hg)")
    chol: float | None = Field(None, ge=0, description="Serum cholesterol (mg/dl)")
    fbs: bool | None = Field(None, description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal["normal", "st-t abnormality", "lv hypertrophy"] | None = Field(
        None, description="Resting electrocardiographic results"
    )
    thalch: float | None = Field(None, ge=0, description="Max heart rate achieved")
    exang: bool | None = Field(None, description="Exercise induced angina")
    oldpeak: float | None = Field(None, description="ST depression induced by exercise")
    slope: Literal["upsloping", "flat", "downsloping"] | None = Field(
        None, description="Slope of the peak exercise ST segment"
    )
    ca: float | None = Field(None, ge=0, le=4, description="# of major vessels (0-3) coloured by flourosopy")
    thal: Literal["normal", "fixed defect", "reversable defect"] | None = Field(
        None, description="Thalassemia result"
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., ge=0, le=1, description="1 = heart disease risk, 0 = no disease")
    label: Literal["disease", "no_disease"]
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the returned label")
    model_version: str
    threshold: float = 0.5


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
