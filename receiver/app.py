import socket
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import numpy as np

# --- CONFIGURATION ---
HOST = '0.0.0.0'
PORT = 65432
IMAGE_DIR = '/app/images'
CLASSES = ['cat', 'car', 'dog']

# --- MODEL SETUP (Same as sender to ensure consistency) ---
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()
preprocess = weights.transforms()

def get_image_feature_vector(image_path):
    """Utility function to extract feature vector."""
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        features = feature_extractor(batch_t)
        return features.squeeze().numpy()

def create_knowledge_base():
    """
    Generates the ideal feature vector for each known class and stores it.
    This is the receiver's semantic "ground truth".
    """
    print("Creating receiver's knowledge base...")
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print("LIST DIR")
    print(files)
    knowledge_base = {}
    for class_name in CLASSES:
        image_path = f"{IMAGE_DIR}/{class_name}.jpeg"
        knowledge_base[class_name] = get_image_feature_vector(image_path)
        print(f" - Generated vector for '{class_name}'")
    return knowledge_base

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def decode_semantic_meaning(noisy_vector, knowledge_base):
    """Finds the best match for the noisy vector from the knowledge base."""
    max_similarity = -1
    best_match = "UNKNOWN"
    for class_name, ideal_vector in knowledge_base.items():
        similarity = cosine_similarity(noisy_vector, ideal_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = class_name
    return best_match, max_similarity

# --- MAIN RECEIVER LOOP ---
knowledge_base = create_knowledge_base()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("\nReceiver is listening for connections...")
    
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"\nConnected by {addr}")
            data = conn.recv(4096)
            if not data:
                continue

            original_label_bytes, noisy_vector_bytes = data.split(b'|', 1)
            original_label = original_label_bytes.decode('utf-8')
            noisy_vector = np.frombuffer(noisy_vector_bytes, dtype=np.float32)
            
            decoded_label, similarity = decode_semantic_meaning(noisy_vector, knowledge_base)
            
            print(f"Original Label: {original_label}")
            print(f"-> Decoded Meaning: **{decoded_label}** (Similarity: {similarity:.4f})")
            if original_label == decoded_label:
                print("✅ SUCCESS: Meaning was recovered despite noise.")
            else:
                print("❌ FAILURE: Meaning was lost due to noise.")