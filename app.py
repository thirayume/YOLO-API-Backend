from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
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

# Configure device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")

# Models configuration
MODELS_DIR = Path("/tmp/models")
MODELS_DIR.mkdir(exist_ok=True)

models = {}

def load_yolov5_local(model_path):
    """Load YOLOv5 model from local file without downloading from GitHub"""
    sys.path.append('ultralytics/yolov5')
    from models.experimental import attempt_load
    model = attempt_load(model_path, device=DEVICE)
    return model

async def load_model(model_version):
    """Load model from local storage"""
    try:
        if model_version not in models:
            model_path = MODELS_DIR / f"{model_version}.pt"
            logger.debug(f"Loading model from: {model_path}")
            
            if not model_path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model file not found: {model_version}.pt"
                )
            
            try:
                if model_version == 'v5':
                    # Load YOLOv5 locally without GitHub
                    models[model_version] = load_yolov5_local(model_path)
                else:
                    # Load other versions using ultralytics
                    models[model_version] = YOLO(str(model_path))
                
                logger.debug(f"Model {model_version} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading model: {str(e)}"
                )
        
        return models[model_version]
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        raise

@app.get("/api/health")
async def health_check():
    model_files = {}
    for model_file in MODELS_DIR.glob("*.pt"):
        model_files[model_file.stem] = {
            "exists": True,
            "size": model_file.stat().st_size,
            "loaded": model_file.stem in models
        }
    
    return {
        "status": "ok",
        "device": DEVICE,
        "models_directory": str(MODELS_DIR),
        "model_files": model_files,
        "models_loaded": list(models.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    try:
        logger.debug(f"Starting prediction request for model {model_version}")
        logger.debug(f"File received: {file.filename}")
        
        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        image = Image.open(io.BytesIO(contents))
        logger.debug(f"Image loaded successfully, size: {image.size}, mode: {image.mode}")
        
        # Load model
        model = await load_model(model_version)
        logger.debug("Model loaded successfully")
        
        # Run prediction
        try:
            results = model(image)
            logger.debug("Prediction completed")
            
            # Process results
            if model_version == 'v5':
                # For YOLOv5 local model
                boxes = []
                if hasattr(results, 'xyxy'):
                    boxes = results.xyxy[0].cpu().numpy().tolist()
                else:
                    pred = results[0] if isinstance(results, list) else results
                    boxes = pred.cpu().numpy().tolist()
            else:
                # For YOLOv8 and newer
                boxes = results[0].boxes.data.cpu().numpy().tolist()
            
            formatted_boxes = [
                {
                    'box': box[:4],
                    'confidence': float(box[4]),
                    'class': int(box[5])
                }
                for box in boxes
            ]
            
            logger.debug(f"Found {len(formatted_boxes)} detections")
            
            return {
                'boxes': formatted_boxes,
                'model_version': model_version,
                'image_size': image.size,
                'num_detections': len(formatted_boxes)
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)