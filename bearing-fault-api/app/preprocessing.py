import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def create_windows(
    X: np.ndarray, 
    window_size: int = 500, 
    step_size: int = 250
) -> np.ndarray:
    """Create sliding windows from time series data."""
    if len(X) < window_size:
        logger.warning(f"Data length ({len(X)}) < window size ({window_size})")
        return np.zeros((0, window_size, X.shape[1]), dtype=np.float32)
    
    windows = []
    for i in range(0, len(X) - window_size + 1, step_size):
        windows.append(X[i:i + window_size])
    
    if len(windows) == 0:
        return np.zeros((0, window_size, X.shape[1]), dtype=np.float32)
    
    return np.array(windows, dtype=np.float32)

def preprocess_for_model(
    df: pd.DataFrame,
    selected_features: List[str],
    window_size: int,
    step_size: int,
    scaler=None,
    return_3d: bool = True
) -> Tuple[np.ndarray, int, int, int]:
    """
    Preprocess data for model prediction.
    
    Args:
        df: Input dataframe
        selected_features: Features to use
        window_size: Window size for sliding windows
        step_size: Step size between windows
        scaler: Optional scaler to apply
        return_3d: If True, return (n_windows, window_size, n_features)
                  If False, return flattened (n_windows, window_size * n_features)
    
    Returns:
        X_processed: Processed data
        n_samples: Number of windows
        T: Window size
        F: Number of features
    """
    # Extract raw data
    X_raw = df[selected_features].values
    
    # Create windows
    X_windows = create_windows(X_raw, window_size, step_size)
    n_samples, T, F = X_windows.shape
    
    if n_samples == 0:
        raise ValueError(f"Not enough data. Need at least {window_size} rows, got {len(X_raw)}")
    
    # Apply scaler if provided
    if scaler is not None:
        if return_3d:
            X_scaled = scaler.transform(X_windows.reshape(-1, F)).reshape(n_samples, T, F)
        else:
            X_flat = X_windows.reshape(n_samples, T * F)
            X_scaled = scaler.transform(X_flat)
            return X_scaled, n_samples, T, F
    else:
        X_scaled = X_windows if return_3d else X_windows.reshape(n_samples, T * F)
    
    return X_scaled, n_samples, T, F

def validate_features(df: pd.DataFrame, required_features: List[str]) -> List[str]:
    """Validate that required features exist in dataframe."""
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return required_features