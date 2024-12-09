from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models dictionary
models = {}

async def load_model(model_version: str):
    if model_version not in models:
        model_url = os.environ.get(f"MODEL_{model_version.upper()}_URL")
        if not model_url:
            raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
        try:
            models[model_version] = YOLO(model_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return models[model_version]

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        model = await load_model(model_version)
        results = model(image)
        
        boxes = results[0].boxes.data.cpu().numpy().tolist()
        
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