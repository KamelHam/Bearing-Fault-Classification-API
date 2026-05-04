from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

class BaseModel(ABC):
    """Base class for all fault classification models."""
    
    def __init__(self, name: str, class_order: List[str]):
        self.name = name
        self.class_order = class_order
        self._model = None
        self._scaler = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self) -> bool:
        """Load the model and scaler."""
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions."""
        pass
    
    def predict_batch(
        self, 
        df: pd.DataFrame,
        selected_features: List[str],
        window_size: int,
        step_size: int
    ) -> Dict[str, Any]:
        """Batch prediction with preprocessing."""
        from app.preprocessing import preprocess_for_model
        
        # Preprocess
        X_processed, n_samples, T, F = preprocess_for_model(
            df, selected_features, window_size, step_size, 
            self._scaler, return_3d=self.expects_3d()
        )
        
        # Predict
        predictions, probabilities = self.predict(X_processed)
        
        # Map to class names
        pred_names = [self.class_order[int(i)] for i in predictions]
        
        return {
            "selected_features": selected_features,
            "num_windows": int(n_samples),
            "window_size": window_size,
            "predictions": pred_names,
            "class_distribution": {cls: pred_names.count(cls) for cls in set(pred_names)},
            "probabilities": probabilities.tolist() if probabilities is not None else None
        }
    
    def expects_3d(self) -> bool:
        """Override for models that expect 3D input (LSTM, CNN, etc.)."""
        return False