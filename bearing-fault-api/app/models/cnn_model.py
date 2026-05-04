import joblib
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import tensorflow as tf

from app.models.base import BaseModel
from app.config import MODEL_PATHS

logger = logging.getLogger(__name__)


class CNNModel(BaseModel):
    """CNN classifier for time series fault detection."""
    
    def __init__(self, name: str, class_order: List[str]):
        super().__init__(name, class_order)
        self.window_size = None
        self.n_features = None
    
    def load(self) -> bool:
        """Load CNN model and scaler."""
        try:
            model_path = MODEL_PATHS["cnn"]["model"]
            scaler_path = MODEL_PATHS["cnn"]["scaler"]
            
            if not Path(model_path).exists():
                logger.error(f"CNN model not found at {model_path}")
                return False
            
            self._model = tf.keras.models.load_model(model_path, compile=False)
            self._scaler = joblib.load(scaler_path)
            
            # CNN expects (batch, timesteps, features)
            self.input_shape = self._model.input_shape
            if len(self.input_shape) == 3:
                self.window_size = self.input_shape[1]
                self.n_features = self.input_shape[2]
            
            self.is_loaded = True
            logger.info(f"CNN model loaded. Window: {self.window_size}, Features: {self.n_features}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            self.is_loaded = False
            return False
    
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with CNN."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure correct shape
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        probabilities = self._model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        if return_proba:
            return predictions, probabilities
        
        return predictions, None
    
    def expects_3d(self) -> bool:
        """CNN expects 3D input."""
        return True