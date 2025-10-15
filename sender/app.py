import socket
import time
import random
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
CHANNEL_HOST = 'channel'
CHANNEL_PORT = 65431
IMAGE_DIR = '/app/images'
IMAGE_FILENAMES = ['cat.jpeg', 'car.jpeg', 'dog.jpeg']

# --- MODEL SETUP ---
# Load a pre-trained ResNet-18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
# We use the model as a feature extractor, so we remove the final classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval() # Set model to evaluation mode

# --- IMAGE PREPROCESSING ---
# The transformations must match what the model was trained on
preprocess = weights.transforms()

def get_image_feature_vector(image_path):
    """Loads an image, preprocesses it, and extracts its feature vector."""
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0) # Create a mini-batch as expected by the model

    with torch.no_grad():
        features = feature_extractor(batch_t)
        # Flatten the features to a 1D vector (ResNet18 gives a 512-element vector)
        vector = features.squeeze().numpy()
    return vector

# --- MAIN SENDER LOOP ---
print("Sender is starting...")
time.sleep(10) # Give other services time to start up

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print("Sender trying to connect to channel...")
            s.connect((CHANNEL_HOST, CHANNEL_PORT))
            print("Sender connected to channel. Starting to send image semantics.")
            
            while True:
                # Pick a random image to send
                image_name = random.choice(IMAGE_FILENAMES)
                image_path = os.path.join(IMAGE_DIR, image_name)
                
                # 1. Get the semantic vector
                vector = get_image_feature_vector(image_path)
                
                # 2. Serialize the vector and label for sending
                # We send the original label for comparison at the receiver
                label = image_name.split('.')[0]
                message = label.encode('utf-8') + b'|' + vector.tobytes()

                print(f"Sending semantics for: {label} (Vector size: {vector.shape[0]})")
                s.sendall(message)
                
                time.sleep(5)

    except Exception as e:
        print(f"An error occurred: {e}. Retrying in 5 seconds...")
        time.sleep(5)