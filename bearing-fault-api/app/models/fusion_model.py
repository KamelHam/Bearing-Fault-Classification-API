import joblib
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
import logging
from pathlib import Path

from app.models.base import BaseModel
from app.config import MODEL_PATHS

logger = logging.getLogger(__name__)


class FusionModel(BaseModel):
    """Ensemble model combining RF and SVM with majority voting."""
    
    def __init__(self, name: str, class_order: List[str]):
        super().__init__(name, class_order)
        self.rf_model = None
        self.svm_model = None
        self.rf_scaler = None
        self.svm_scaler = None
        self.expected_features = None
    
    def load(self) -> bool:
        """Load both RF and SVM models and scalers."""
        try:
            # Load RF
            rf_model_path = MODEL_PATHS["rf"]["model"]
            rf_scaler_path = MODEL_PATHS["rf"]["scaler"]
            
            if not Path(rf_model_path).exists():
                logger.error(f"RF model for fusion not found at {rf_model_path}")
                return False
            
            self.rf_model = joblib.load(rf_model_path)
            self.rf_scaler = joblib.load(rf_scaler_path)
            
            # Load SVM
            svm_model_path = MODEL_PATHS["svm"]["model"]
            svm_scaler_path = MODEL_PATHS["svm"]["scaler"]
            
            if not Path(svm_model_path).exists():
                logger.error(f"SVM model for fusion not found at {svm_model_path}")
                return False
            
            self.svm_model = joblib.load(svm_model_path)
            self.svm_scaler = joblib.load(svm_scaler_path)
            
            # Use RF's expected features as reference
            self.expected_features = self.rf_scaler.n_features_in_
            self.is_loaded = True
            
            logger.info(f"Fusion model loaded. RF+SVM ensemble ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")
            self.is_loaded = False
            return False
    
    def majority_vote(self, predictions: List[int]) -> int:
        """Perform majority voting on predictions."""
        vote_counts = Counter(predictions)
        majority = max(vote_counts.items(), key=lambda x: x[1])[0]
        return majority
    
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using majority voting between RF and SVM."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Get predictions from both models
        rf_preds = self.rf_model.predict(X)
        svm_preds = self.svm_model.predict(X)
        
        # Apply majority voting for each sample
        final_predictions = np.array([
            self.majority_vote([rf_preds[i], svm_preds[i]]) 
            for i in range(len(X))
        ])
        
        if return_proba:
            # Approximate probabilities from voting
            probabilities = np.zeros((len(X), len(self.class_order)))
            for i in range(len(X)):
                votes = [rf_preds[i], svm_preds[i]]
                for vote in votes:
                    probabilities[i, int(vote)] += 0.5
            return final_predictions, probabilities
        
        return final_predictions, None
    
    def predict_batch(
        self, 
        df, selected_features, window_size, step_size
    ):
        """Override batch prediction to handle fusion-specific scaling."""
        from app.preprocessing import preprocess_for_model
        
        # Preprocess with RF scaler (use as reference)
        X_processed, n_samples, T, F = preprocess_for_model(
            df, selected_features, window_size, step_size, 
            self.rf_scaler, return_3d=False
        )
        
        # Make predictions
        predictions, probabilities = self.predict(X_processed, return_proba=True)
        
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
        """Fusion model expects flattened 2D input."""
        return False