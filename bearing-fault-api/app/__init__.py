"""
Bearing Fault Classification API Package
=========================================

A comprehensive API for induction motor bearing fault classification
using multiple deep learning and machine learning models.

Available models:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)
- Transformer
- Temporal Convolutional Network (TCN)
- Ensemble Fusion (RF + SVM)

"""

__version__ = "2.0.0"
__author__ = "Bearing Fault Detection Team"
__description__ = "Multi-model API for bearing fault classification"

# Package exports
from app.config import (
    CLASS_ORDER,
    ALL_FEATURES,
    MODEL_CONFIGS,
    MODEL_PATHS,
    ModelInfo
)

from app.schemas import (
    ModelType,
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

from app.models.base import BaseModel
from app.models import (
    RFModel,
    SVMModel,
    MLPModel,
    LSTMModel,
    CNNModel,
    TransformerModel,
    TCNModel,
    FusionModel
)

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Config exports
    "CLASS_ORDER",
    "ALL_FEATURES",
    "MODEL_CONFIGS",
    "MODEL_PATHS",
    "ModelInfo",
    
    # Schema exports
    "ModelType",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    
    # Model exports
    "BaseModel",
    "RFModel",
    "SVMModel",
    "MLPModel",
    "LSTMModel",
    "CNNModel",
    "TransformerModel",
    "TCNModel",
    "FusionModel",
]

# Optional: Configure package-level logging
import logging

# Create null handler for the package
# This allows users to configure logging as they wish
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# Set up package logger
logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())
logger.propagate = True

def setup_logging(level=logging.INFO):
    """
    Setup logging for the package.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    
    # Also set level for child loggers
    for name in ['app.models', 'app.preprocessing', 'app.main']:
        child_logger = logging.getLogger(name)
        child_logger.addHandler(handler)
        child_logger.setLevel(level)

def get_package_info() -> dict:
    """
    Get package information.
    
    Returns:
        Dictionary with package metadata
    """
    return {
        "name": "bearing-fault-api",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "models_available": [
            "rf", "svm", "mlp", "lstm", "cnn", "transformer", "tcn", "fusion"
        ],
        "num_classes": len(CLASS_ORDER),
        "classes": CLASS_ORDER,
        "features": ALL_FEATURES
    }