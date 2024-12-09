from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
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

# Initialize models dictionary
models = {}

def get_model_url(model_version):
    """Get model URL from environment variables"""
    url = os.environ.get(f"MODEL_{model_version.upper()}_URL")
    if not url:
        raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
    return url

async def load_model(model_version):
    """Load model if not already loaded"""
    if model_version not in models:
        url = get_model_url(model_version)
        try:
            models[model_version] = YOLO(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return models[model_version]

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_available": ["v5", "v8", "v10", "v11"]
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    """Prediction endpoint"""
    try:
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Load and run model
        model = await load_model(model_version)
        results = model(image)
        
        # Process results
        boxes = results[0].boxes.data.cpu().numpy().tolist()
        
        # Format boxes for consistent output
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
            'model_version': model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))