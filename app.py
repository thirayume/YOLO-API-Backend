from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
import requests
import tempfile
import logging
import traceback
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(tempfile.gettempdir()) / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Your model URLs
MODEL_URLS = {
    'v5': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v5.pt",
    'v8': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v8.pt",
    'v10': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v10.pt",
    'v11': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v11.pt",
}

models = {}

def download_model(model_version):
    """Download model from GitHub if not exists"""
    try:
        model_path = MODELS_DIR / f"{model_version}.pt"
        if not model_path.exists():
            logger.info(f"Starting download of model {model_version}")
            url = MODEL_URLS.get(model_version)
            if not url:
                raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
            
            logger.info(f"Downloading from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        downloaded += len(chunk)
                        f.write(chunk)
                        progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                        logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Model downloaded successfully to {model_path}")
            
            # Verify file exists and has size
            if not model_path.exists() or model_path.stat().st_size == 0:
                raise Exception("Model file is empty or not downloaded correctly")
        
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def load_model(model_version):
    """Load model from local storage"""
    try:
        if model_version not in models:
            logger.info(f"Loading model {model_version}")
            model_path = download_model(model_version)
            
            if model_version == 'v5':
                logger.info("Using YOLOv5 loader")
                models[model_version] = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                     path=str(model_path), force_reload=True)
            else:
                logger.info("Using YOLO loader")
                from ultralytics import YOLO
                models[model_version] = YOLO(str(model_path))
            
            logger.info(f"Model {model_version} loaded successfully")
        return models[model_version]
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/api/health")
async def health_check():
    logger.info("Health check requested")
    models_status = {}
    for model_version in MODEL_URLS.keys():
        model_path = MODELS_DIR / f"{model_version}.pt"
        models_status[model_version] = {
            "downloaded": model_path.exists(),
            "file_size": model_path.stat().st_size if model_path.exists() else 0,
            "loaded": model_version in models
        }
    
    return {
        "status": "ok",
        "available_models": list(MODEL_URLS.keys()),
        "models_loaded": list(models.keys()),
        "models_directory": str(MODELS_DIR),
        "models_status": models_status
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.info(f"Starting prediction request for model {model_version}")
        logger.info(f"File received: {file.filename}")
        
        # Read and validate image
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes")
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Load image
        try:
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image loaded successfully, size: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Load model
        try:
            model = await load_model(model_version)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        # Run prediction
        try:
            logger.info("Running prediction")
            results = model(image)
            logger.info("Prediction completed")
            
            # Process results based on model version
            if model_version == 'v5':
                boxes = results.xyxy[0].cpu().numpy().tolist()
            else:
                boxes = results[0].boxes.data.cpu().numpy().tolist()
            
            logger.info(f"Found {len(boxes)} detections")
            
            # Format boxes
            formatted_boxes = [
                {
                    'box': box[:4],
                    'confidence': float(box[4]),
                    'class': int(box[5])
                }
                for box in boxes
            ]
            
            return {
                'boxes': formatted_boxes,
                'model_version': model_version,
                'image_size': image.size,
                'num_detections': len(formatted_boxes)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))