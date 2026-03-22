from pydantic import BaseModel
from typing import Dict, List


class PredictionResponse(BaseModel):
    filename: str | None = None
    top1: str
    confidence: float
    top3: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]