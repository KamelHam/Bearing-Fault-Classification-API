from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    RF = "rf"
    SVM = "svm"
    MLP = "mlp"
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    TCN = "tcn"
    FUSION = "fusion"

class PredictionRequest(BaseModel):
    features: Optional[List[str]] = Field(
        default=None,
        description="Features to use for prediction"
    )
    model: ModelType = Field(
        default=ModelType.RF,
        description="Model to use for prediction"
    )

class PredictionResponse(BaseModel):
    status: str
    model_used: str
    selected_features: List[str]
    num_windows: int
    window_size: int
    predictions: List[str]  # Changed from predictions_preview to predictions
    class_distribution: Dict[str, int]
    confidence_stats: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    available_endpoints: List[str]