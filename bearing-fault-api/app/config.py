from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# Model paths (you'll need to update these with your actual paths)
MODEL_PATHS = {
    "rf": {
        "model": MODELS_DIR / "final_rf_model.pkl",
        "scaler": MODELS_DIR / "rf_scaler.pkl",
    },
    "svm": {
        "model": MODELS_DIR / "final_svm_model_PROPER.joblib",
        "scaler": MODELS_DIR / "svm_scaler_PROPER.joblib",
    },
    "mlp": {
        "model": MODELS_DIR / "mlp_best_model_PROPER.h5",
        "scaler": MODELS_DIR / "mlp_best_scaler_PROPER.pkl",
    },
    "lstm": {
        "model": MODELS_DIR / "final_lstm_model_PROPER.keras",
        "scaler": MODELS_DIR / "lstm_scaler_PROPER.pkl",

    },
    "cnn": {
        "model": MODELS_DIR / "final_cnn_model_PROPER.h5",
        "scaler": MODELS_DIR / "cnn_scaler_PROPER.pkl",
    },
    "transformer": {
        "model": MODELS_DIR / "final_transformer_model.keras",
        "scaler": MODELS_DIR / "transformer_scaler.pkl",
        "encoder": MODELS_DIR / "final_transformer_model.pkl",
    },
    "tcn": {
        "model": MODELS_DIR / "final_tcn_model.keras",
        "scaler": MODELS_DIR / "tcn_scaler.pkl",
    },
}

# Common configurations
CLASS_ORDER = [
    'HNL', 'FNL', 'FCNL', 'FSNL', 'FRNL',
    'H50L', 'F50L', 'FC50L', 'FS50L', 'FR50L',
    'H100L', 'F100L', 'FC100L', 'FS100L', 'FR100L'
]

ALL_FEATURES = ['RMSFC_Voltage', 'Current_Phase1', 'Current_Phase2', 'Rotor_Speed']

# Model-specific configurations
MODEL_CONFIGS = {
    "rf": {"step_size": 35, "window_size": 280 // len(ALL_FEATURES)},
    "svm": {"step_size": 10, "window_size": 240 // len(ALL_FEATURES)},
    "mlp": {"step_size": 400, "window_size": 800 // len(ALL_FEATURES)},
    "lstm": {"step_size": 100, "window_size": 100},
    "cnn": {"step_size": 250, "window_size": 500},
    "transformer": {"step_size": 250, "window_size": 500},
    "tcn": {"step_size": 50, "window_size": 100},
    "fusion": {"step_size": 250, "window_size": 2000 // len(ALL_FEATURES)},
}

class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    available: bool = Field(..., description="Whether model is loaded")
    window_size: int = Field(..., description="Window size for this model")
    step_size: int = Field(..., description="Step size for windowing")
    expected_features: Optional[int] = Field(None, description="Expected number of features")
    expects_3d: bool = Field(False, description="Whether model expects 3D input")

def get_model_config(model_name: str) -> Dict[str, int]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, {"window_size": 500, "step_size": 250})
