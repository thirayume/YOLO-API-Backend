from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Model URL
MODEL_URLS = {
    'v5': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v5.pt",
    'v8': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v8.pt",
    'v10': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v10.pt",
    'v11': "https://huggingface.co/thirayume/ai-snaily-yolo/resolve/main/v11.pt"
}

# Initialize models dictionary
models = {}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        # Load model if not already loaded
        if model_version not in models:
            if model_version not in MODEL_URLS:
                raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
            models[model_version] = YOLO(MODEL_URLS[model_version])

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run prediction
        results = models[model_version](image)
        
        # Process results
        boxes = results[0].boxes.data.cpu().numpy().tolist()
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