import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os
import logging
from utils.models import PretrainedResNetEncoder
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeEncoder")

app = FastAPI()

# Global model variable
encoder = None
preprocess = None

def load_model():
    global encoder, preprocess
    logger.info("Initializing Edge Encoder (ResNet50/Edge)...")
    encoder = PretrainedResNetEncoder(encoded_space_dim=512)
    
    weights_path = "/app/models/resnet_encoder.pth"
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            encoder.load_state_dict(state_dict)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    else:
        logger.warning("No weights found! Using random/imagenet initialization.")
    
    encoder.eval()
    
    # ResNet expects 224x224
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

@app.on_event("startup")
async def startup_event():
    load_model()



@app.post("/encode")
async def encode_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            vector = encoder(batch_t)
            vector_np = vector.squeeze().numpy().tolist()
            
        return {"vector": vector_np}
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return {"error": str(e)}
