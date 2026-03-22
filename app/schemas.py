from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    top1: str
    confidence: float
    top3: Dict[str, float]
