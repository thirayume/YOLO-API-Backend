import torch
from PIL import Image
import io
import os
from ultralytics import YOLO

# Initialize models dictionary
models = {}

def get_model_urls():
    """Get model URLs from environment variables"""
    return {
        'v5': os.environ.get('MODEL_V5_URL'),
        'v8': os.environ.get('MODEL_V8_URL'),
        'v10': os.environ.get('MODEL_V10_URL'),
        'v11': os.environ.get('MODEL_V11_URL')
    }

async def load_model(model_version):
    """Load model if not already loaded"""
    if model_version not in models:
        model_urls = get_model_urls()
        if model_version not in model_urls:
            raise ValueError(f"Model {model_version} not found")
            
        url = model_urls[model_version]
        if 'v5' in model_version.lower():
            models[model_version] = torch.hub.load('ultralytics/yolov5', 'custom', path=url)
        else:
            models[model_version] = YOLO(url)
            
    return models[model_version]

async def process_image(file, model_version):
    """Process image with specified model"""
    # Read and prepare image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Load and run model
    model = await load_model(model_version)
    results = model(image)
    
    # Process results
    if 'v5' in model_version.lower():
        boxes = results.xyxy[0].cpu().numpy().tolist()
    else:
        boxes = results[0].boxes.data.cpu().numpy().tolist()
    
    # Format boxes for consistent output
    formatted_boxes = []
    for box in boxes:
        formatted_boxes.append({
            'box': box[:4],  # x1, y1, x2, y2
            'confidence': float(box[4]),
            'class': int(box[5])
        })
    
    return {
        'boxes': formatted_boxes,
        'model_version': model_version
    }