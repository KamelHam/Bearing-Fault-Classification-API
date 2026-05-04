import joblib
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.models import load_model

from app.models.base import BaseModel
from app.config import MODEL_PATHS

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    """MLP (Multi-Layer Perceptron) classifier for fault detection."""
    
    def __init__(self, name: str, class_order: List[str]):
        super().__init__(name, class_order)
        self.expected_features = None
        self.input_shape = None
    
    def load(self) -> bool:
        """Load MLP model and scaler."""
        try:
            model_path = MODEL_PATHS["mlp"]["model"]
            scaler_path = MODEL_PATHS["mlp"]["scaler"]
            
            if not Path(model_path).exists():
                logger.error(f"MLP model not found at {model_path}")
                return False
            
            self._model = load_model(model_path, compile=False)
            self._scaler = joblib.load(scaler_path)
            
            # Get expected input shape
            self.input_shape = self._model.input_shape
            self.expected_features = self.input_shape[-1]
            self.is_loaded = True
            
            logger.info(f"MLP model loaded. Expected features: {self.expected_features}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MLP model: {e}")
            self.is_loaded = False
            return False
    
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with MLP."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure correct shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        probabilities = self._model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        if return_proba:
            return predictions, probabilities
        
        return predictions, None
    
    def expects_3d(self) -> bool:
        """MLP expects flattened 2D input."""
        return False