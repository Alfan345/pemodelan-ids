# Quick Start Guide - IDS Middleware API

## Installation

```bash
cd middleware-ids
pip install -r requirements.txt
```

## Copy Model Artifacts

```bash
# Copy from the artifacts_multiclass_collapsed_v5 directory
cp -r ../artifacts_multiclass_collapsed_v5/* artifacts/
```

## Run the Server

```bash
# Development mode
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Test the API

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single prediction
data = {
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

response = requests.post("http://localhost:8000/api/v1/predict", json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Attack: {result['is_attack']}")
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/api/v1/model/info

# Get required features
curl http://localhost:8000/api/v1/features

# Make a prediction (save your data in request.json)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t ids-middleware:latest .

# Run container
docker run -d \
  --name ids-api \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  ids-middleware:latest
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_api.py::TestHealthEndpoints::test_health_check -v
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Model Not Loading
- Ensure artifacts folder contains all required files:
  - config.json
  - label_map.json
  - model_state.pt
  - report.json
  - scaler.pkl
  - transform_meta.json

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: requires Python 3.8+

### Port Already in Use
- Change the port: `uvicorn app.main:app --port 8001`
- Or kill the existing process using the port

### Performance Issues
- Use multiple workers: `uvicorn app.main:app --workers 4`
- Consider using gunicorn: `gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`
