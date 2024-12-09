from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import torch
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
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

# Model URLs
MODEL_URLS = {
    'v5': "https://drive.google.com/uc?id=14RIqgapBW838rYClk99mKZh49dZN3AsI",
    'v8': "https://drive.google.com/uc?id=14zV33EmnBNj9U137OUwWA1iLOzTQEI-C",
    'v10': "https://drive.google.com/uc?id=15I-DjAgqBiYqTKockdgJpH9Z9JrJ6RTq",
    'v11': "https://drive.google.com/uc?id=15V3b0kVWVvvvkV5Uhzjb5gDeCtiaOSRK"
}

# Initialize models dictionary
models = {}

def load_yolov5(url):
    """Load YOLOv5 model"""
    try:
        logger.info(f"Loading YOLOv5 model from {url}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=url)
        logger.info("YOLOv5 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {str(e)}")
        raise

def load_yolov8_plus(url):
    """Load YOLOv8 and newer models"""
    try:
        logger.info(f"Loading YOLO model from {url}")
        model = YOLO(url)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        raise

async def load_model(model_version):
    """Load appropriate model based on version"""
    try:
        if model_version not in models:
            if model_version not in MODEL_URLS:
                logger.error(f"Model version {model_version} not found")
                raise HTTPException(status_code=404, detail=f"Model {model_version} not found")
                
            url = MODEL_URLS[model_version]
            if model_version == 'v5':
                models[model_version] = load_yolov5(url)
            else:
                models[model_version] = load_yolov8_plus(url)
        
        return models[model_version]
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "available_models": list(MODEL_URLS.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.info(f"Received prediction request for model {model_version}")
        logger.info(f"File received: {file.filename}")
        
        # Verify file content
        if not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # Read and validate image
        contents = await file.read()
        if not contents:
            logger.error("Empty file received")
            raise HTTPException(status_code=400, detail="Empty file received")
        
        try:
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image opened successfully: {image.size}")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error opening image: {str(e)}")
        
        # Load model
        model = await load_model(model_version)
        logger.info("Model loaded successfully")
        
        # Run prediction
        results = model(image)
        logger.info("Prediction completed")
        
        # Process results based on model version
        if model_version == 'v5':
            boxes = results.xyxy[0].cpu().numpy().tolist()
        else:
            boxes = results[0].boxes.data.cpu().numpy().tolist()
        
        logger.info(f"Number of detections: {len(boxes)}")
        
        # Format boxes consistently
        formatted_boxes = []
        for box in boxes:
            formatted_boxes.append({
                'box': box[:4],
                'confidence': float(box[4]),
                'class': int(box[5])
            })
        
        return JSONResponse({
            'boxes': formatted_boxes,
            'model_version': model_version,
            'image_size': image.size,
            'num_detections': len(formatted_boxes)
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)