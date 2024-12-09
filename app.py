from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model URLs
MODEL_URLS = {
    'v5': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v5.pt",
    'v8': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v8.pt",
    'v10': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v10.pt",
    'v11': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v11.pt"
}

# Initialize models dictionary
models = {}

def load_yolov5(url):
    """Load YOLOv5 model"""
    return torch.hub.load('ultralytics/yolov5', 'custom', path=url)

def load_yolov8_plus(url):
    """Load YOLOv8 and newer models"""
    return YOLO(url)

async def load_model(model_version):
    """Load appropriate model based on version"""
    if model_version not in models:
        if model_version not in MODEL_URLS:
            raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
            
        url = MODEL_URLS[model_version]
        try:
            if model_version == 'v5':
                models[model_version] = load_yolov5(url)
            else:
                models[model_version] = load_yolov8_plus(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return models[model_version]

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "available_models": list(MODEL_URLS.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        # Load and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get appropriate model
        model = await load_model(model_version)
        
        # Run prediction
        results = model(image)
        
        # Process results based on model version
        if model_version == 'v5':
            boxes = results.xyxy[0].cpu().numpy().tolist()
        else:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
        
        # Format boxes consistently
        formatted_boxes = []
        for box in boxes:
            formatted_boxes.append({
                'box': box[:4],
                'confidence': float(box[4]),
                'class': int(box[5])
            })
        
        return {
            'boxes': formatted_boxes,
            'model_version': model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))