"""
DNN model for intrusion detection.
"""
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from .preprocessing import FlowPreprocessor


class DNNClassifier(nn.Module):
    """Deep Neural Network classifier for IDS."""
    
    def __init__(self, input_dim: int = 50, num_classes: int = 5, 
                 layers: List[int] = [512, 256, 128], dropout: float = 0.35):
        """
        Initialize the DNN classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(DNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers dynamically
        layer_list = []
        prev_size = input_dim
        
        for layer_size in layers:
            layer_list.append(nn.Linear(prev_size, layer_size))
            layer_list.append(nn.LeakyReLU())
            layer_list.append(nn.Dropout(dropout))
            prev_size = layer_size
        
        # Output layer
        layer_list.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layer_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class IDSModel:
    """Wrapper for the IDS model with preprocessing and inference."""
    
    def __init__(self, artifacts_path: Path):
        """
        Initialize the IDS model.
        
        Args:
            artifacts_path: Path to the artifacts directory
        """
        self.artifacts_path = Path(artifacts_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configurations
        self.config = self._load_config()
        self.label_map = self._load_label_map()
        self.report = self._load_report()
        
        # Initialize preprocessor
        self.preprocessor = FlowPreprocessor(artifacts_path)
        
        # Initialize and load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        config_path = self.artifacts_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_label_map(self) -> Dict:
        """Load label mapping."""
        label_map_path = self.artifacts_path / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map file not found at {label_map_path}")
        
        with open(label_map_path, "r") as f:
            return json.load(f)
    
    def _load_report(self) -> Dict:
        """Load model performance report."""
        report_path = self.artifacts_path / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Report file not found at {report_path}")
        
        with open(report_path, "r") as f:
            return json.load(f)
    
    def _load_model(self) -> DNNClassifier:
        """Load the trained model."""
        model_path = self.artifacts_path / "model_state.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model state file not found at {model_path}")
        
        # Initialize model with config parameters
        model = DNNClassifier(
            input_dim=self.config.get("input_dim", 50),
            num_classes=self.config.get("num_classes", 5),
            layers=self.config.get("layers", [512, 256, 128]),
            dropout=self.config.get("dropout", 0.35)
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Fix state dict keys if needed (handle both formats)
        # Old format: "0.weight", "3.weight", etc.
        # New format: "network.0.weight", "network.3.weight", etc.
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("network."):
                # Add "network." prefix if missing
                new_key = f"network.{key}"
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        
        return model
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict a single sample.
        
        Args:
            features: Dictionary of feature names to values
            
        Returns:
            Tuple of (prediction, confidence, probabilities_dict)
        """
        # Validate features
        is_valid, missing = self.preprocessor.validate_features(features)
        if not is_valid:
            raise ValueError(f"Missing required features: {missing}")
        
        # Preprocess
        processed = self.preprocessor.transform(features)
        
        # Convert to tensor
        x = torch.FloatTensor(processed).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item()
        
        # Get label
        prediction = self.label_map["id_to_label"][str(pred_idx)]
        
        # Create probabilities dict
        probs_dict = {
            self.label_map["id_to_label"][str(i)]: probabilities[0, i].item()
            for i in range(len(self.label_map["id_to_label"]))
        }
        
        return prediction, confidence, probs_dict
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Predict multiple samples.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of tuples (prediction, confidence, probabilities_dict)
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information and performance metrics."""
        return {
            "config": self.config,
            "label_map": self.label_map,
            "performance": self.report,
            "required_features": self.preprocessor.get_feature_names()
        }
