import joblib
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from app.models.base import BaseModel
from app.config import MODEL_PATHS, ALL_FEATURES

logger = logging.getLogger(__name__)


class RFModel(BaseModel):
    """Random Forest classifier for fault detection."""
    
    def __init__(self, name: str, class_order: List[str]):
        super().__init__(name, class_order)
        self.expected_features = None
    
    def load(self) -> bool:
        """Load Random Forest model and scaler."""
        try:
            model_path = MODEL_PATHS["rf"]["model"]
            scaler_path = MODEL_PATHS["rf"]["scaler"]
            
            if not Path(model_path).exists():
                logger.error(f"RF model not found at {model_path}")
                return False
            
            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            self.expected_features = self._scaler.n_features_in_
            self.is_loaded = True
            
            logger.info(f"RF model loaded. Expected features: {self.expected_features}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RF model: {e}")
            self.is_loaded = False
            return False
    
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with Random Forest."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        predictions = self._model.predict(X)
        
        if return_proba:
            probabilities = self._model.predict_proba(X)
            return predictions, probabilities
        
        return predictions, None
    
    def expects_3d(self) -> bool:
        """RF expects flattened 2D input."""
        return False