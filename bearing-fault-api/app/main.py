from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import logging

from app.config import CLASS_ORDER, ALL_FEATURES, MODEL_CONFIGS, ModelInfo
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelType
from app.preprocessing import validate_features
from app.models.rf_model import RFModel
from app.models.svm_model import SVMModel
from app.models.mlp_model import MLPModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model import CNNModel
from app.models.transformer_model import TransformerModel
from app.models.tcn_model import TCNModel
from app.models.fusion_model import FusionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Bearing Fault Classification API",
    description="Multi-model API for induction motor bearing fault classification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize models
models = {
    ModelType.RF: RFModel("Random Forest", CLASS_ORDER),
    ModelType.SVM: SVMModel("SVM", CLASS_ORDER),
    ModelType.MLP: MLPModel("MLP", CLASS_ORDER),
    ModelType.LSTM: LSTMModel("LSTM", CLASS_ORDER),
    ModelType.CNN: CNNModel("CNN", CLASS_ORDER),
    ModelType.TRANSFORMER: TransformerModel("Transformer", CLASS_ORDER),
    ModelType.TCN: TCNModel("TCN", CLASS_ORDER),
    ModelType.FUSION: FusionModel("RF+SVM Fusion", CLASS_ORDER),
}

# Load all models on startup
@app.on_event("startup")
async def load_models():
    """Load all models on startup."""
    logger.info("Loading models...")
    for model_name, model in models.items():
        try:
            model.load()
            logger.info(f"✅ Loaded {model_name.value} model")
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name.value}: {e}")
    logger.info("Model loading complete")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health and available models."""
    return {
        "status": "healthy",
        "models_loaded": {name.value: model.is_loaded for name, model in models.items()},
        "available_endpoints": [
            "/predict",
            "/predict/{model_name}",
            "/models/info",
            "/features",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models": {name.value: {
            "loaded": model.is_loaded,
            "window_size": MODEL_CONFIGS.get(name.value, {}).get("window_size", 500),
            "step_size": MODEL_CONFIGS.get(name.value, {}).get("step_size", 250)
        } for name, model in models.items()},
        "available_features": ALL_FEATURES,
        "num_classes": len(CLASS_ORDER),
        "classes": CLASS_ORDER
    }

@app.get("/models/info")
async def get_models_info():
    """Get information about all available models."""
    return {
        "models": [
            {
                "name": name.value,
                "loaded": model.is_loaded,
                "window_size": MODEL_CONFIGS.get(name.value, {}).get("window_size", 500),
                "step_size": MODEL_CONFIGS.get(name.value, {}).get("step_size", 250),
                "expects_3d": model.expects_3d()
            }
            for name, model in models.items()
        ],
        "default_model": ModelType.RF.value,
        "features": ALL_FEATURES,
        "classes": CLASS_ORDER
    }

@app.get("/features")
async def get_features():
    """Get available features."""
    return {
        "available_features": ALL_FEATURES,
        "descriptions": {
            "RMSFC_Voltage": "Root mean square of voltage",
            "Current_Phase1": "Current measurement for phase 1",
            "Current_Phase2": "Current measurement for phase 2",
            "Rotor_Speed": "Rotor rotational speed"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Excel file with sensor data"),
    model_name: ModelType = Query(default=ModelType.RF, description="Model to use"),
    features: Optional[List[str]] = Query(default=None, description="Features to use")
):
    """
    Predict bearing faults using the specified model.
    
    - Upload an Excel file with sensor data
    - Select which model to use (rf, svm, mlp, lstm, cnn, transformer, tcn, fusion)
    - Optionally specify which features to use
    - Returns predictions with class distribution
    """
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "File must be an Excel file (.xlsx or .xls)")
    
    # Get model
    model = models.get(model_name)
    if not model or not model.is_loaded:
        raise HTTPException(404, f"Model {model_name} not available")
    
    try:
        # Read file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        logger.info(f"Processing {file.filename} with shape {df.shape}")
        
        # Select features
        selected_features = features or ALL_FEATURES
        selected_features = validate_features(df, selected_features)
        
        # Get model config
        config = MODEL_CONFIGS.get(model_name.value, {"window_size": 500, "step_size": 250})
        
        # Predict
        result = model.predict_batch(
            df, selected_features, 
            config["window_size"], config["step_size"]
        )
        
        return PredictionResponse(
            status="success",
            model_used=model_name.value,
            **result
        )
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_specific_model(
    model_name: ModelType,
    file: UploadFile = File(...),
    features: Optional[List[str]] = Query(default=None)
):
    """Predict using a specific model (path parameter version)."""
    return await predict(file, model_name, features)

@app.post("/predict/all")
async def predict_all_models(
    file: UploadFile = File(...),
    features: Optional[List[str]] = Query(default=None)
):
    """
    Predict using all available models and return comparison.
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "File must be an Excel file")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        selected_features = features or ALL_FEATURES
        selected_features = validate_features(df, selected_features)
        
        results = {}
        for name, model in models.items():
            if model.is_loaded:
                config = MODEL_CONFIGS.get(name.value, {"window_size": 500, "step_size": 250})
                try:
                    result = model.predict_batch(
                        df, selected_features,
                        config["window_size"], config["step_size"]
                    )
                    results[name.value] = {
                        "class_distribution": result["class_distribution"],
                        "num_windows": result["num_windows"]
                    }
                except Exception as e:
                    results[name.value] = {"error": str(e)}
        
        return {
            "status": "success",
            "selected_features": selected_features,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)