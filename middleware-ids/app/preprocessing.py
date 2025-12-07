"""
Preprocessing pipeline for network flow features.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Union


class FlowPreprocessor:
    """Preprocessor for network flow features."""
    
    def __init__(self, artifacts_path: Path):
        """
        Initialize the preprocessor.
        
        Args:
            artifacts_path: Path to the artifacts directory
        """
        self.artifacts_path = Path(artifacts_path)
        self.scaler = None
        self.transform_meta = None
        self.cols = None
        self.heavy_cols = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load scaler and transform metadata."""
        # Load scaler
        scaler_path = self.artifacts_path / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load transform metadata
        transform_meta_path = self.artifacts_path / "transform_meta.json"
        if not transform_meta_path.exists():
            raise FileNotFoundError(f"Transform meta file not found at {transform_meta_path}")
        
        with open(transform_meta_path, "r") as f:
            self.transform_meta = json.load(f)
        
        self.cols = self.transform_meta.get("cols", [])
        self.heavy_cols = self.transform_meta.get("heavy_cols", [])
        
        if not self.cols:
            raise ValueError("No columns found in transform_meta.json")
    
    def validate_features(self, features: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        Validate that all required features are present.
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = [col for col in self.cols if col not in features]
        return len(missing) == 0, missing
    
    def transform(self, features: Union[Dict[str, float], pd.DataFrame]) -> np.ndarray:
        """
        Transform features using the preprocessing pipeline.
        
        Steps:
        1. Select columns from transform_meta
        2. Apply log1p transform to heavy_cols
        3. Scale using loaded scaler
        
        Args:
            features: Dictionary or DataFrame of feature values
            
        Returns:
            Transformed numpy array ready for model input
        """
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features.copy()
        
        # Select required columns in the correct order
        df = df[self.cols]
        
        # Apply log1p transformation to heavy columns
        for col in self.heavy_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        # Scale the features
        scaled = self.scaler.transform(df)
        
        return scaled
    
    def get_feature_names(self) -> List[str]:
        """Get the list of required feature names."""
        return self.cols.copy()
