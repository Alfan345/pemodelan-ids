"""
Pydantic schemas for request and response validation.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """Request schema for single prediction."""
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Destination Port": 80.0,
                "Flow Duration": 120000.0,
                "Total Fwd Packets": 8.0,
                "Total Length of Fwd Packets": 1200.0,
                "Fwd Packet Length Max": 200.0,
                "Fwd Packet Length Min": 100.0,
                "Fwd Packet Length Mean": 150.0,
                "Fwd Packet Length Std": 30.0,
                "Bwd Packet Length Max": 180.0,
                "Bwd Packet Length Min": 90.0,
                "Bwd Packet Length Mean": 135.0,
                "Flow Bytes/s": 10000.0,
                "Flow Packets/s": 66.6,
                "Flow IAT Mean": 15000.0,
                "Flow IAT Std": 5000.0,
                "Flow IAT Max": 30000.0,
                "Flow IAT Min": 5000.0,
                "Fwd IAT Mean": 17000.0,
                "Fwd IAT Std": 4500.0,
                "Fwd IAT Min": 6000.0,
                "Bwd IAT Total": 105000.0,
                "Bwd IAT Mean": 15000.0,
                "Bwd IAT Std": 4000.0,
                "Bwd IAT Max": 28000.0,
                "Bwd IAT Min": 7000.0,
                "Fwd PSH Flags": 0.0,
                "Fwd URG Flags": 0.0,
                "Fwd Header Length": 160.0,
                "Bwd Header Length": 140.0,
                "Bwd Packets/s": 50.0,
                "Min Packet Length": 90.0,
                "Max Packet Length": 200.0,
                "Packet Length Mean": 145.0,
                "Packet Length Variance": 1500.0,
                "FIN Flag Count": 1.0,
                "RST Flag Count": 0.0,
                "PSH Flag Count": 2.0,
                "ACK Flag Count": 10.0,
                "URG Flag Count": 0.0,
                "Down/Up Ratio": 0.875,
                "Init_Win_bytes_forward": 65535.0,
                "Init_Win_bytes_backward": 65535.0,
                "act_data_pkt_fwd": 5.0,
                "min_seg_size_forward": 20.0,
                "Active Mean": 30000.0,
                "Active Std": 5000.0,
                "Active Max": 40000.0,
                "Active Min": 20000.0,
                "Idle Mean": 50000.0,
                "Idle Std": 10000.0
            }
        }
    )
    
    # Required 50 features from CIC-IDS2017 dataset
    destination_port: float = Field(..., alias="Destination Port")
    flow_duration: float = Field(..., alias="Flow Duration")
    total_fwd_packets: float = Field(..., alias="Total Fwd Packets")
    total_length_of_fwd_packets: float = Field(..., alias="Total Length of Fwd Packets")
    fwd_packet_length_max: float = Field(..., alias="Fwd Packet Length Max")
    fwd_packet_length_min: float = Field(..., alias="Fwd Packet Length Min")
    fwd_packet_length_mean: float = Field(..., alias="Fwd Packet Length Mean")
    fwd_packet_length_std: float = Field(..., alias="Fwd Packet Length Std")
    bwd_packet_length_max: float = Field(..., alias="Bwd Packet Length Max")
    bwd_packet_length_min: float = Field(..., alias="Bwd Packet Length Min")
    bwd_packet_length_mean: float = Field(..., alias="Bwd Packet Length Mean")
    flow_bytes_s: float = Field(..., alias="Flow Bytes/s")
    flow_packets_s: float = Field(..., alias="Flow Packets/s")
    flow_iat_mean: float = Field(..., alias="Flow IAT Mean")
    flow_iat_std: float = Field(..., alias="Flow IAT Std")
    flow_iat_max: float = Field(..., alias="Flow IAT Max")
    flow_iat_min: float = Field(..., alias="Flow IAT Min")
    fwd_iat_mean: float = Field(..., alias="Fwd IAT Mean")
    fwd_iat_std: float = Field(..., alias="Fwd IAT Std")
    fwd_iat_min: float = Field(..., alias="Fwd IAT Min")
    bwd_iat_total: float = Field(..., alias="Bwd IAT Total")
    bwd_iat_mean: float = Field(..., alias="Bwd IAT Mean")
    bwd_iat_std: float = Field(..., alias="Bwd IAT Std")
    bwd_iat_max: float = Field(..., alias="Bwd IAT Max")
    bwd_iat_min: float = Field(..., alias="Bwd IAT Min")
    fwd_psh_flags: float = Field(..., alias="Fwd PSH Flags")
    fwd_urg_flags: float = Field(..., alias="Fwd URG Flags")
    fwd_header_length: float = Field(..., alias="Fwd Header Length")
    bwd_header_length: float = Field(..., alias="Bwd Header Length")
    bwd_packets_s: float = Field(..., alias="Bwd Packets/s")
    min_packet_length: float = Field(..., alias="Min Packet Length")
    max_packet_length: float = Field(..., alias="Max Packet Length")
    packet_length_mean: float = Field(..., alias="Packet Length Mean")
    packet_length_variance: float = Field(..., alias="Packet Length Variance")
    fin_flag_count: float = Field(..., alias="FIN Flag Count")
    rst_flag_count: float = Field(..., alias="RST Flag Count")
    psh_flag_count: float = Field(..., alias="PSH Flag Count")
    ack_flag_count: float = Field(..., alias="ACK Flag Count")
    urg_flag_count: float = Field(..., alias="URG Flag Count")
    down_up_ratio: float = Field(..., alias="Down/Up Ratio")
    init_win_bytes_forward: float = Field(..., alias="Init_Win_bytes_forward")
    init_win_bytes_backward: float = Field(..., alias="Init_Win_bytes_backward")
    act_data_pkt_fwd: float = Field(..., alias="act_data_pkt_fwd")
    min_seg_size_forward: float = Field(..., alias="min_seg_size_forward")
    active_mean: float = Field(..., alias="Active Mean")
    active_std: float = Field(..., alias="Active Std")
    active_max: float = Field(..., alias="Active Max")
    active_min: float = Field(..., alias="Active Min")
    idle_mean: float = Field(..., alias="Idle Mean")
    idle_std: float = Field(..., alias="Idle Std")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with original feature names."""
        return self.model_dump(by_alias=True)


class PredictResponse(BaseModel):
    """Response schema for single prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "BENIGN",
                "confidence": 0.9987,
                "is_attack": False,
                "probabilities": {
                    "BENIGN": 0.9987,
                    "Brute Force": 0.0003,
                    "DDoS": 0.0005,
                    "DoS": 0.0003,
                    "Port Scan": 0.0002
                }
            }
        }
    )
    
    prediction: str = Field(..., description="Predicted attack class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    is_attack: bool = Field(..., description="Whether traffic is malicious")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "flows": [
                    {
                        "Destination Port": 80.0,
                        "Flow Duration": 120000.0,
                        # ... other features
                    }
                ]
            }
        }
    )
    
    flows: List[Dict[str, float]] = Field(..., description="List of flow features")


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "prediction": "BENIGN",
                        "confidence": 0.9987,
                        "is_attack": False,
                        "probabilities": {"BENIGN": 0.9987}
                    }
                ],
                "total": 1
            }
        }
    )
    
    predictions: List[PredictResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config": {
                    "layers": [512, 256, 128],
                    "activation": "leaky_relu",
                    "dropout": 0.35
                },
                "performance": {
                    "macro_f1": 0.9790,
                    "accuracy": 0.9953
                },
                "labels": ["BENIGN", "Brute Force", "DDoS", "DoS", "Port Scan"],
                "input_features": 50
            }
        }
    )
    
    config: Dict = Field(..., description="Model configuration")
    performance: Dict = Field(..., description="Model performance metrics")
    labels: List[str] = Field(..., description="Available class labels")
    input_features: int = Field(..., description="Number of input features")


class FeaturesResponse(BaseModel):
    """Response schema for feature list."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": ["Destination Port", "Flow Duration", "..."],
                "count": 50
            }
        }
    )
    
    features: List[str] = Field(..., description="List of required feature names")
    count: int = Field(..., description="Number of features")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }
    )
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
