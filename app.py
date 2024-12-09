from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
import gdown
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

MODEL_IDS = {
    'v5': '14RIqgapBW838rYClk99mKZh49dZN3AsI',
    'v8': '14zV33EmnBNj9U137OUwWA1iLOzTQEI-C',
    'v10': '15I-DjAgqBiYqTKockdgJpH9Z9JrJ6RTq',
    'v11': '15V3b0kVWVvvvkV5Uhzjb5gDeCtiaOSRK'
}

models = {}

def download_model(model_version):
    """Download model from Google Drive if not exists"""
    model_path = MODELS_DIR / f"{model_version}.pt"
    if not model_path.exists():
        logger.info(f"Downloading model {model_version}")
        file_id = MODEL_IDS.get(model_version)
        if not file_id:
            raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
        
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(model_path), quiet=False)
    
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
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    return models[model_version]

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "available_models": list(MODEL_IDS.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.info(f"Received prediction request for model {model_version}")
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
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
        
        return {
            'boxes': formatted_boxes,
            'model_version': model_version,
            'image_size': image.size
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)