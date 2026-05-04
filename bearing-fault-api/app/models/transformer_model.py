import joblib
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

from app.models.base import BaseModel
from app.config import MODEL_PATHS

logger = logging.getLogger(__name__)


class TransformerModel(BaseModel):
    """Transformer classifier for time series fault detection."""
    
    def __init__(self, name: str, class_order: List[str]):
        super().__init__(name, class_order)
        self.window_size = None
        self.n_features = None
        self.label_encoder = None
        self.expected_timesteps = 120  # Your transformer expects 120 timesteps
    
    def load(self) -> bool:
        """Load Transformer model, scaler, and label encoder."""
        try:
            model_path = MODEL_PATHS["transformer"]["model"]
            scaler_path = MODEL_PATHS["transformer"]["scaler"]
            encoder_path = MODEL_PATHS["transformer"].get("encoder")
            
            # Try to get encoder path from config
            if not encoder_path:
                encoder_path = Path(model_path.parent / "transformer_label_encoder.pkl")
            
            if not Path(model_path).exists():
                logger.error(f"Transformer model not found at {model_path}")
                return False
            
            # Load model without compilation for faster inference
            self._model = load_model(model_path, compile=False)
            self._scaler = joblib.load(scaler_path)
            
            # Load label encoder if exists
            if Path(encoder_path).exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"Label encoder loaded with {len(self.label_encoder.classes_)} classes")
            
            # Get input shape
            self.input_shape = self._model.input_shape
            if len(self.input_shape) == 3:
                self.window_size = self.input_shape[1]
                self.n_features = self.input_shape[2]
            else:
                # Use default if shape not as expected
                self.window_size = self.expected_timesteps
                self.n_features = 4
            
            self.is_loaded = True
            logger.info(f"Transformer model loaded. Window: {self.window_size}, Features: {self.n_features}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_input(self, df, selected_features, window_size, step_size):
        """Preprocess data specifically for Transformer model."""
        from app.preprocessing import preprocess_for_model
        
        # For Transformer that expects fixed timesteps
        X_raw = df[selected_features].values
        
        # Calculate number of samples based on fixed timesteps
        samples = len(X_raw) // self.expected_timesteps
        if samples == 0:
            raise ValueError(f"Not enough data rows ({len(X_raw)}) for {self.expected_timesteps}-step windowing.")
        
        # Take exact number of samples
        X = X_raw[:samples * self.expected_timesteps].reshape(
            samples, self.expected_timesteps, len(selected_features)
        )
        
        # Scale the data
        X_scaled = self._scaler.transform(
            X.reshape(-1, len(selected_features))
        ).reshape(X.shape)
        
        return X_scaled, samples, self.expected_timesteps, len(selected_features)
    
    def predict(
        self, 
        X: np.ndarray, 
        return_proba: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with Transformer."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure correct shape (samples, timesteps, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], self.expected_timesteps, -1)
        
        probabilities = self._model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        if return_proba:
            return predictions, probabilities
        
        return predictions, None
    
    def predict_batch(
        self, 
        df, selected_features, window_size, step_size
    ) -> dict:
        """Batch prediction with Transformer-specific preprocessing."""
        # Preprocess using Transformer's specific method
        X_processed, n_samples, T, F = self.preprocess_input(
            df, selected_features, window_size, step_size
        )
        
        # Predict
        predictions, probabilities = self.predict(X_processed, return_proba=True)
        
        # Map to class names using label encoder if available
        if self.label_encoder is not None:
            pred_names = self.label_encoder.inverse_transform(predictions).tolist()
        else:
            pred_names = [self.class_order[int(i)] for i in predictions]
        
        # Calculate confidence statistics
        confidences = np.max(probabilities, axis=1) if probabilities is not None else np.ones(len(predictions)) * 0.85
        
        return {
            "selected_features": selected_features,
            "num_windows": int(n_samples),
            "window_size": T,
            "predictions": pred_names,
            "predictions_preview": pred_names[:50],
            "class_distribution": {cls: pred_names.count(cls) for cls in set(pred_names)},
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "confidence_stats": {
                "average": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "minimum": float(np.min(confidences)),
                "maximum": float(np.max(confidences))
            }
        }
    
    def expects_3d(self) -> bool:
        """Transformer expects 3D input."""
        return True