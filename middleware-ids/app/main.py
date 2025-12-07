"""
FastAPI application for IDS middleware.
"""
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model import IDSModel
from .schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    FeaturesResponse,
    HealthResponse
)

# Global model instance
model: IDSModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading."""
    global model
    
    # Startup: Load model
    try:
        print(f"Loading model from {settings.artifacts_path}")
        model = IDSModel(settings.artifacts_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="REST API for Intrusion Detection System using Deep Learning",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "IDS Middleware API",
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version=settings.version
    )


@app.get(
    f"{settings.api_prefix}/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"]
)
async def get_model_info():
    """Get model information and performance metrics."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = model.get_model_info()
    
    return ModelInfoResponse(
        config=info["config"],
        performance=info["performance"],
        labels=list(info["label_map"]["label_to_id"].keys()),
        input_features=len(info["required_features"])
    )


@app.get(
    f"{settings.api_prefix}/features",
    response_model=FeaturesResponse,
    tags=["Model"]
)
async def get_features():
    """Get list of required input features."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    features = model.preprocessor.get_feature_names()
    
    return FeaturesResponse(
        features=features,
        count=len(features)
    )


@app.post(
    f"{settings.api_prefix}/predict",
    response_model=PredictResponse,
    tags=["Prediction"]
)
async def predict_single(request: PredictRequest):
    """
    Predict attack type for a single network flow.
    
    Accepts 50 network flow features and returns the predicted attack class.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert request to dict
        features = request.to_dict()
        
        # Make prediction
        prediction, confidence, probabilities = model.predict_single(features)
        
        # Determine if it's an attack
        is_attack = prediction != "BENIGN"
        
        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            is_attack=is_attack,
            probabilities=probabilities
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    f"{settings.api_prefix}/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"]
)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict attack types for multiple network flows.
    
    Accepts a list of network flows and returns predictions for each.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if not request.flows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No flows provided"
        )
    
    try:
        # Make predictions
        results = model.predict_batch(request.flows)
        
        # Format responses
        predictions = []
        for prediction, confidence, probabilities in results:
            is_attack = prediction != "BENIGN"
            predictions.append(
                PredictResponse(
                    prediction=prediction,
                    confidence=confidence,
                    is_attack=is_attack,
                    probabilities=probabilities
                )
            )
        
        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions)
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Exception handler for better error messages
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
