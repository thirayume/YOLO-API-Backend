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
import gc  # For garbage collection
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
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

# Ensure CUDA is available if possible
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

MODELS_DIR = Path(tempfile.gettempdir()) / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    'v5': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v5.pt",
    'v8': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v8.pt",
    'v10': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v10.pt",
    'v11': "https://github.com/thirayume/YOLO-API-Backend/releases/download/ai-snaily-v1/v11.pt",
}

models = {}

def clean_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def download_model(model_version):
    """Download model from GitHub if not exists"""
    try:
        model_path = MODELS_DIR / f"{model_version}.pt"
        if not model_path.exists():
            logger.debug(f"Starting download of model {model_version}")
            url = MODEL_URLS.get(model_version)
            if not url:
                raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
            
            # Download with timeout and retry
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.debug(f"Download completed. File size: {model_path.stat().st_size} bytes")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Download failed: {str(e)}")
                if model_path.exists():
                    model_path.unlink()  # Remove partial download
                raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")
        
        if not model_path.exists() or model_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Model file is empty or not downloaded correctly")
        
        return model_path
    except Exception as e:
        logger.error(f"Error in download_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def load_model(model_version):
    """Load model from local storage"""
    try:
        if model_version not in models:
            logger.debug(f"Loading model {model_version}")
            model_path = download_model(model_version)
            
            # Clean memory before loading new model
            clean_memory()
            
            try:
                if model_version == 'v5':
                    logger.debug("Using YOLOv5 loader")
                    models[model_version] = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                         path=str(model_path), force_reload=True)
                    models[model_version].to(DEVICE)
                else:
                    logger.debug("Using YOLO loader")
                    from ultralytics import YOLO
                    models[model_version] = YOLO(str(model_path))
                
                logger.debug(f"Model {model_version} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                if model_version in models:
                    del models[model_version]
                clean_memory()
                raise
        
        return models[model_version]
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.debug(f"Starting prediction request for model {model_version}")
        logger.debug(f"File received: {file.filename}")
        
        # Read image
        try:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Empty file received")
            
            image = Image.open(io.BytesIO(contents))
            logger.debug(f"Image loaded successfully, size: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")
        
        # Load model
        try:
            model = await load_model(model_version)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
        
        # Run prediction
        try:
            logger.debug("Running prediction")
            results = model(image)
            logger.debug("Prediction completed")
            
            # Process results
            if model_version == 'v5':
                boxes = results.xyxy[0].cpu().numpy().tolist()
            else:
                boxes = results[0].boxes.data.cpu().numpy().tolist()
            
            formatted_boxes = [
                {
                    'box': box[:4],
                    'confidence': float(box[4]),
                    'class': int(box[5])
                }
                for box in boxes
            ]
            
            logger.debug(f"Processed {len(formatted_boxes)} detections")
            
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
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        clean_memory()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up after prediction
        clean_memory()

@app.get("/api/health")
async def health_check():
    try:
        return {
            "status": "ok",
            "device": DEVICE,
            "cuda_available": torch.cuda.is_available(),
            "models_directory": str(MODELS_DIR),
            "models_loaded": list(models.keys()),
            "available_models": list(MODEL_URLS.keys())
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {"status": "error", "detail": str(e)}