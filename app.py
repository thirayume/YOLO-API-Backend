from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
import requests
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a models directory in /tmp
MODELS_DIR = Path(tempfile.gettempdir()) / "models"
MODELS_DIR.mkdir(exist_ok=True)

# GitHub release URLs for models
MODEL_URLS = {
    'v5': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v5.pt",
    'v8': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v8.pt",
    'v10': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v10.pt",
    'v11': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v11.pt",
}

models = {}

def download_model(model_version):
    """Download model from GitHub if not exists"""
    model_path = MODELS_DIR / f"{model_version}.pt"
    if not model_path.exists():
        logger.info(f"Downloading model {model_version}")
        url = MODEL_URLS.get(model_version)
        if not url:
            raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded successfully to {model_path}")
    
    return model_path

async def load_model(model_version):
    """Load model from local storage"""
    if model_version not in models:
        try:
            model_path = download_model(model_version)
            if model_version == 'v5':
                models[model_version] = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                     path=str(model_path))
            else:
                from ultralytics import YOLO
                models[model_version] = YOLO(str(model_path))
            logger.info(f"Model {model_version} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    return models[model_version]

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "available_models": list(MODEL_URLS.keys()),
        "models_loaded": list(models.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.info(f"Received prediction request for model {model_version}")
        logger.info(f"Processing file: {file.filename}")
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image loaded successfully, size: {image.size}")
        
        # Load model
        model = await load_model(model_version)
        
        # Run prediction
        results = model(image)
        
        # Process results based on model version
        if model_version == 'v5':
            boxes = results.xyxy[0].cpu().numpy().tolist()
        else:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
        
        # Format boxes
        formatted_boxes = [
            {
                'box': box[:4],
                'confidence': float(box[4]),
                'class': int(box[5])
            }
            for box in boxes
        ]
        
        logger.info(f"Found {len(formatted_boxes)} detections")
        
        return {
            'boxes': formatted_boxes,
            'model_version': model_version,
            'image_size': image.size,
            'num_detections': len(formatted_boxes)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))