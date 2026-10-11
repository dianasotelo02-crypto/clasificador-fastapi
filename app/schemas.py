"""Esquemas de entrada/salida. Pydantic valida los datos automáticamente."""
from typing import List
from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Largo del sépalo (cm)", examples=[5.1])
    sepal_width: float = Field(..., gt=0, le=10, description="Ancho del sépalo (cm)", examples=[3.5])
    petal_length: float = Field(..., gt=0, le=10, description="Largo del pétalo (cm)", examples=[1.4])
    petal_width: float = Field(..., gt=0, le=10, description="Ancho del pétalo (cm)", examples=[0.2])


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    samples: List[IrisFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
