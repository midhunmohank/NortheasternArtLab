from fastapi import FastAPI, Depends, HTTPException, status
import requests
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel,SecretStr
from typing import Dict
import os

app = FastAPI()
DIFFUSION_API_KEY = os.environ.get("DIFFUSION_API_KEY")
class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = None
    width: str = "512"
    height: str = "512"
    prompt_strength: float = 1.0
    samples: str = "1"
    num_inference_steps: str = "20"
    seed: str = None
    guidance_scale: float = 7.5
    safety_checker: str = "yes"
    webhook: str = None
    track_id: str = None

@app.post("/text2img")
def text2img(request: TextToImageRequest):
    url = "https://stablediffusionapi.com/api/v3/text2img"

    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "key": DIFFUSION_API_KEY,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "prompt_strength": request.prompt_strength,
        "samples": request.samples,
        "num_inference_steps": request.num_inference_steps,
        "seed": request.seed,
        "guidance_scale": request.guidance_scale,
        "safety_checker": request.safety_checker,
        "webhook": request.webhook,
        "track_id": request.track_id
    }
    response = requests.post(url, headers=headers, data=payload)
    return response
    # if response.status_code == 200:
    #     data = response.json()
    #     if data["status"] == "success":
    #         return {"output": data["output"]}
    #     if data["status"] == "processing":
    #         return {"output": data["eta"], "fetch_url": data["fetch_result"]}
    # else:
    #     raise HTTPException(status_code=400, detail="Error")
