from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Return mock response for testing
        return {
            'boxes': [
                {
                    'box': [100, 100, 200, 200],
                    'confidence': 0.95,
                    'class': 0
                }
            ],
            'model_version': model_version
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))