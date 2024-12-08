from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .utils import process_image, get_model_urls

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_urls = get_model_urls()
    return {
        "status": "ok",
        "available_models": list(model_urls.keys())
    }

@app.post("/api/predict/{model_version}")
async def predict(model_version: str, file: UploadFile = File(...)):
    """Prediction endpoint"""
    try:
        return await process_image(file, model_version)
    except Exception as e:
        return {"error": str(e)}, 500