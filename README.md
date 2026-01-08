# UCSense-AI

An AI-powered Graph-to-Data pipeline for digitizing geotechnical UCS (Unconfined Compressive Strength) test results with 95%+ accuracy.

## Features

- **Image Rectification**: Perspective correction and deskewing using OpenCV
- **Signal Extraction**: Morphological filtering to isolate UCS curves from noise
- **OCR Mapping**: Tesseract-based axis detection with intelligent correction
- **Curve Extraction**: Hybrid approach using contour tracing and U-Net segmentation
- **Feature Engineering**: Automatic peak UCS and failure strain detection
- **High Accuracy**: Confidence scoring with Savitzky-Golay smoothing

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, OpenCV, Tesseract OCR, TensorFlow
- **Frontend**: Next.js 14, React 18, Recharts
- **Database**: Supabase (PostgreSQL)
- **Infrastructure**: AWS Lambda, S3, API Gateway

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
├── backend/                # FastAPI application
│   ├── main.py            # Application entry point
│   ├── config.py          # Configuration settings
│   ├── preprocessing/     # Image rectification & filtering
│   ├── ocr/               # Axis detection & coordinate mapping
│   ├── extraction/        # Curve extraction (OpenCV + U-Net)
│   ├── analysis/          # Feature engineering & peak detection
│   ├── ml/                # Machine learning models
│   ├── routers/           # API endpoints
│   ├── models/            # Pydantic schemas
│   └── utils/             # Helper functions
├── frontend/              # Next.js application
├── infra/                 # AWS/Docker configurations
├── tests/                 # Test suites
├── data/                  # Sample graphs & ground truth
└── models/                # Trained model weights
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/extract` | POST | Upload image for extraction |
| `/api/v1/extract/{job_id}` | GET | Get extraction results |
| `/api/v1/extract/{job_id}/csv` | GET | Download data as CSV |
| `/api/v1/validate` | POST | Validate against ground truth |
| `/health` | GET | Health check |

## License

MIT License
