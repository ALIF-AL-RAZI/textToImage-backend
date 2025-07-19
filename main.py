from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI(title="Text-to-Image API (HF Inference)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://www.alifalrazi.com/projectslist/textToImage"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API configuration
HF_TOKEN = os.getenv('HF_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

class ImageRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 3.5
    num_inference_steps: int = 50

class ImageResponse(BaseModel):
    image_base64: str
    prompt: str

def query_hf_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    try:
        # Prepare payload for HF API
        payload = {
            "inputs": request.prompt,
            "parameters": {
                "height": request.height,
                "width": request.width,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps
            }
        }
        
        response = query_hf_api(payload)
        
        if response.status_code == 503:
            raise HTTPException(status_code=503, detail="Model is loading, please wait...")
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # Convert response to base64
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode()
        
        return ImageResponse(
            image_base64=image_base64,
            prompt=request.prompt
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if HF_TOKEN else "no_token",
        "message": "Using Hugging Face Inference API" if HF_TOKEN else "HF_TOKEN required"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)