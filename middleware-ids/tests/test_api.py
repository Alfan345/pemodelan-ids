"""
Test cases for IDS API.
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.sample_data import SAMPLE_BENIGN, SAMPLE_DDOS, SAMPLE_PORTSCAN, SAMPLE_INCOMPLETE


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["model_loaded"] == True


class TestModelEndpoints:
    """Test model information endpoints."""
    
    def test_get_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "performance" in data
        assert "labels" in data
        assert "input_features" in data
        assert data["input_features"] == 50
        assert len(data["labels"]) == 5
    
    def test_get_features(self, client):
        """Test features list endpoint."""
        response = client.get("/api/v1/features")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "count" in data
        assert data["count"] == 50
        assert len(data["features"]) == 50


class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_single_benign(self, client):
        """Test single prediction with benign traffic."""
        response = client.post("/api/v1/predict", json=SAMPLE_BENIGN)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "is_attack" in data
        assert "probabilities" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) == 5
    
    def test_predict_single_ddos(self, client):
        """Test single prediction with DDoS traffic."""
        response = client.post("/api/v1/predict", json=SAMPLE_DDOS)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "is_attack" in data
        assert "probabilities" in data
    
    def test_predict_single_portscan(self, client):
        """Test single prediction with port scan traffic."""
        response = client.post("/api/v1/predict", json=SAMPLE_PORTSCAN)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "is_attack" in data
        assert "probabilities" in data
    
    def test_predict_batch(self, client):
        """Test batch prediction."""
        batch_data = {
            "flows": [SAMPLE_BENIGN, SAMPLE_DDOS, SAMPLE_PORTSCAN]
        }
        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
        
        # Check each prediction
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred
            assert "is_attack" in pred
            assert "probabilities" in pred
    
    def test_predict_missing_features(self, client):
        """Test prediction with missing features."""
        response = client.post("/api/v1/predict", json=SAMPLE_INCOMPLETE)
        # Should fail validation
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty list."""
        batch_data = {"flows": []}
        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/api/v1/invalid")
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test using wrong HTTP method."""
        response = client.get("/api/v1/predict")
        assert response.status_code == 405  # Method Not Allowed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
