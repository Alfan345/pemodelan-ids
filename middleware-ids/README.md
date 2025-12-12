# IDS Middleware API

REST API middleware untuk sistem deteksi intrusi (Intrusion Detection System) berbasis Deep Learning menggunakan model DNN yang dilatih pada dataset CIC-IDS2017.

## Deskripsi

API ini menyediakan endpoint untuk klasifikasi traffic jaringan secara real-time menggunakan model Deep Neural Network (DNN). Model dapat mengklasifikasikan traffic ke dalam 5 kategori:
- **BENIGN**: Traffic normal/aman
- **Brute Force**: Serangan brute force
- **DDoS**: Serangan Distributed Denial of Service
- **DoS**: Serangan Denial of Service
- **Port Scan**: Aktivitas port scanning

## Performa Model

Model yang digunakan memiliki performa excellent pada dataset CIC-IDS2017:

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.53% |
| **Macro F1-Score** | 97.90% |
| **Weighted F1-Score** | 99.54% |

### Performa Per Kelas

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BENIGN | 99.98% | 99.46% | 99.72% | 644,036 |
| Brute Force | 86.78% | 99.96% | 92.91% | 2,745 |
| DDoS | 99.84% | 99.92% | 99.88% | 38,404 |
| DoS | 95.58% | 99.86% | 97.67% | 58,124 |
| Port Scan | 98.74% | 99.88% | 99.30% | 27,208 |

## Arsitektur Model

- **Input**: 50 fitur dari network flow (CIC-IDS2017)
- **Hidden Layers**: 
  - Layer 1: 512 neurons + LeakyReLU + Dropout(0.35)
  - Layer 2: 256 neurons + LeakyReLU + Dropout(0.35)
  - Layer 3: 128 neurons + LeakyReLU + Dropout(0.35)
- **Output**: 5 classes (Softmax)
- **Framework**: PyTorch 2.1.0

## Quick Start

### Prerequisites

- Python 3.8+
- Model artifacts (harus di-copy ke folder `artifacts/`)

### Instalasi

1. Clone repository:
```bash
git clone <repository-url>
cd middleware-ids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy model artifacts:
```bash
# Copy artifacts dari folder artifacts_multiclass_collapsed_v5
cp -r ../artifacts_multiclass_collapsed_v5/* artifacts/
```

4. Jalankan server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server akan berjalan di `http://localhost:8000`

### Akses Dokumentasi API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

#### GET `/health`
Check status kesehatan API dan model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Model Information

#### GET `/api/v1/model/info`
Mendapatkan informasi model dan performa.

**Response:**
```json
{
  "config": {
    "layers": [512, 256, 128],
    "activation": "leaky_relu",
    "dropout": 0.35,
    "input_dim": 50,
    "num_classes": 5
  },
  "performance": {
    "macro_f1": 0.9790,
    "accuracy": 0.9953,
    "weighted_f1": 0.9954
  },
  "labels": ["BENIGN", "Brute Force", "DDoS", "DoS", "Port Scan"],
  "input_features": 50
}
```

#### GET `/api/v1/features`
Mendapatkan daftar 50 fitur yang dibutuhkan.

**Response:**
```json
{
  "features": [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    ...
  ],
  "count": 50
}
```

### Prediction

#### POST `/api/v1/predict`
Prediksi single network flow.

**Request Body:**
```json
{
  "Destination Port": 80.0,
  "Flow Duration": 120000.0,
  "Total Fwd Packets": 8.0,
  "Total Length of Fwd Packets": 1200.0,
  ...
}
```

**Response:**
```json
{
  "prediction": "BENIGN",
  "confidence": 0.9987,
  "is_attack": false,
  "probabilities": {
    "BENIGN": 0.9987,
    "Brute Force": 0.0003,
    "DDoS": 0.0005,
    "DoS": 0.0003,
    "Port Scan": 0.0002
  }
}
```

#### POST `/api/v1/predict/batch`
Prediksi multiple network flows sekaligus.

**Request Body:**
```json
{
  "flows": [
    {
      "Destination Port": 80.0,
      "Flow Duration": 120000.0,
      ...
    },
    {
      "Destination Port": 443.0,
      "Flow Duration": 150000.0,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "BENIGN",
      "confidence": 0.9987,
      "is_attack": false,
      "probabilities": {...}
    },
    {
      "prediction": "DDoS",
      "confidence": 0.9821,
      "is_attack": true,
      "probabilities": {...}
    }
  ],
  "total": 2
}
```

## Input Features

API membutuhkan 50 fitur dari network flow (sesuai CIC-IDS2017):

### Packet-level Features
- Destination Port
- Flow Duration
- Total Fwd/Bwd Packets
- Fwd/Bwd Packet Length (Max, Min, Mean, Std)
- Packet Length (Min, Max, Mean, Variance)

### Flow-level Features
- Flow Bytes/s
- Flow Packets/s
- Flow IAT (Mean, Std, Max, Min)
- Fwd/Bwd IAT (Total, Mean, Std, Max, Min)

### Header Features
- Fwd/Bwd Header Length
- Flag Counts (FIN, RST, PSH, ACK, URG)
- Fwd PSH/URG Flags

### TCP Features
- Init_Win_bytes_forward/backward
- act_data_pkt_fwd
- min_seg_size_forward

### Timing Features
- Active (Mean, Std, Max, Min)
- Idle (Mean, Std)

### Other Features
- Down/Up Ratio
- Bwd Packets/s

Lihat `/api/v1/features` untuk daftar lengkap dengan format yang tepat.

## Docker Deployment

### Build Image
```bash
docker build -t ids-middleware:latest .
```

### Run Container
```bash
docker run -d \
  --name ids-api \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  ids-middleware:latest
```

### Using Docker Compose
```bash
docker-compose up -d
```

## Testing

Run tests menggunakan pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Development

### Project Structure
```
middleware-ids/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model loading & inference
│   ├── preprocessing.py     # Feature preprocessing
│   ├── schemas.py           # Pydantic schemas
│   └── config.py            # Configuration
├── artifacts/               # Model artifacts
│   ├── model_state.pt
│   ├── config.json
│   ├── label_map.json
│   ├── report.json
│   ├── scaler.pkl
│   └── transform_meta.json
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── sample_data.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Environment Variables

Buat file `.env` untuk konfigurasi:

```env
API_PREFIX=/api/v1
APP_NAME=IDS Middleware API
VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
RELOAD=False
```

## Example Usage

### Python Client
```python
import requests

# Single prediction
url = "http://localhost:8000/api/v1/predict"
data = {
    "Destination Port": 80.0,
    "Flow Duration": 120000.0,
    # ... other 48 features
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Attack: {result['is_attack']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Preprocessing Pipeline

1. **Feature Selection**: Memilih 50 fitur sesuai `transform_meta.json`
2. **Log Transform**: Menerapkan `log1p()` pada fitur-fitur "heavy"
3. **Scaling**: Normalisasi menggunakan RobustScaler
4. **Inference**: Forward pass melalui DNN model

## Performance Tips

- Gunakan **batch prediction** untuk multiple flows (lebih efisien)
- Deploy dengan Gunicorn/Uvicorn workers untuk production
- Gunakan GPU jika tersedia (otomatis terdeteksi)
- Cache model di memory (sudah di-handle oleh lifespan)

## Troubleshooting

### Model tidak load
- Pastikan folder `artifacts/` berisi semua file yang dibutuhkan
- Cek log error saat startup
- Verify path di `config.py`

### Prediction error
- Validasi bahwa semua 50 fitur ada
- Cek format data (harus float/numeric)
- Pastikan nama fitur exact match (case-sensitive, spasi)

### Performance lambat
- Gunakan batch prediction untuk multiple requests
- Deploy dengan multiple workers
- Pertimbangkan GPU acceleration

## License

[License Information]

## Authors

- [Your Name/Team]

## References

- CIC-IDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- FastAPI: https://fastapi.tiangolo.com/
- PyTorch: https://pytorch.org/
