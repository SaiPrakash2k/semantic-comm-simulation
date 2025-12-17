
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import logging
from utils.models import PretrainedMobileNetEncoder, SimpleDecoder, PretrainedResNetEncoder, ComplexDecoder
from tqdm import tqdm # For progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(encoder, decoder, name, train_loader, epochs=5, learning_rate=1e-3, device='cpu'):
    logging.info(f"--- Starting training for {name} ---")
    encoder.to(device)
    decoder.to(device)
    
    criterion = nn.MSELoss()
    # OPTIMIZE: Only optimize decoder parameters + encoder's final layer (if trainable)
    # The Backbone is frozen in __init__, so filter(lambda p: p.requires_grad, ...) works.
    params_to_optimize = list(decoder.parameters()) + list(filter(lambda p: p.requires_grad, encoder.parameters()))
    
    optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
    
    # Encoder in eval mode (BatchNorm stats should stay frozen usually, or maybe not?)
    # For Transfer Learning, usually we keep backbone in eval mode.
    encoder.eval() 
    decoder.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        # Wrap loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)
            
            # Forward pass
            with torch.set_grad_enabled(True):
                # We can compute encoder output
                # If backbone is frozen, autograd graph stops at the last layer.
                features = encoder(images)
                outputs = decoder(features)
                loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"[{name}] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
    return encoder, decoder

def main():
    # Configuration
    EPOCHS = 3 # Convergence should be fast with pretrained
    BATCH_SIZE = 32
    LR = 1e-3
    SAVE_DIR = './models'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Data
    logging.info("Loading CIFAR-10...")
    # MobileNet/ResNet expect normalized images typically, but we'll stick to [0,1] for simplicity of reconstruction target.
    # The models might not be optimal without mean/std norm but they are robust.
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # MobileNet/ResNet prefer 224. 32x32 is too small for deep nets.
        transforms.ToTensor()
    ])
    # However, our system uses 32x32. Resizing to 224 slows things down MASSIVELY and might send HUGE raw data?
    # NO, we only resize for the ENCODER input. The RAW action sends original 32x32.
    # WAIT: SimpleDecoder outputs 32x32. If we resize input to 224, we reconstruct 32x32?
    # Yes, we calculate loss against original 32x32 (or resized target).
    # Let's resize input to 224 for Encoder, but target is 32x32?
    # SimpleDecoder outputs 32x32.
    # So: Input -> Resize(224) -> Encoder -> Vector -> Decoder -> Reconstructed(32) -> Loss(Original 32).
    # Correct.
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    
    # We need raw images for target? No, we can just use another transform or resize back.
    # Let's just create a dataset that returns (img_224, img_32).
    # Or just load dataset twice?
    # Simpler: Load 32x32. Resize on the fly inside the loop if needed.
    # CIFAR is 32x32 natively.
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    resize_224 = transforms.Resize((224, 224))

    # 2. Train Local (MobileNetV3 + SimpleDecoder)
    logging.info("Training Local Node (MobileNetV3)...")
    mobilenet = PretrainedMobileNetEncoder(encoded_space_dim=512)
    # Important: MobileNet expects 224x224. 
    # Logic Update: In the loop, we will resize.
    
    simple_decoder = SimpleDecoder(encoded_space_dim=512)
    
    # Custom training loop adaptation
    mobilenet.to(device)
    simple_decoder.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(simple_decoder.parameters()) + list(filter(lambda p: p.requires_grad, mobilenet.parameters())), lr=LR)
    
    mobilenet.eval()
    simple_decoder.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[MobileNet+SimpleDec] Epoch {epoch+1}/{EPOCHS}")
        for images_32, _ in pbar:
            images_32 = images_32.to(device)
            images_224 = resize_224(images_32) # Resize for encoder
            
            features = mobilenet(images_224)
            reconstructed_32 = simple_decoder(features) # Output is 32x32
            
            loss = criterion(reconstructed_32, images_32) # Compare with original 32x32
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images_32.size(0)
            pbar.set_postfix({'loss': loss.item()})
        logging.info(f"[MobileNet+SimpleDec] Epoch {epoch+1}, Loss: {running_loss/len(train_loader.dataset):.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save separately
    torch.save(mobilenet.state_dict(), os.path.join(SAVE_DIR, "mobilenet_encoder.pth"))
    torch.save(simple_decoder.state_dict(), os.path.join(SAVE_DIR, "simple_decoder.pth"))


    # 3. Train Edge (ResNet50 + ComplexDecoder)
    logging.info("Training Edge Node (ResNet50)...")
    resnet = PretrainedResNetEncoder(encoded_space_dim=512)
    complex_decoder = ComplexDecoder(encoded_space_dim=512)
    
    resnet.to(device)
    complex_decoder.to(device)
    optimizer = optim.Adam(list(complex_decoder.parameters()) + list(filter(lambda p: p.requires_grad, resnet.parameters())), lr=LR)
    
    resnet.eval()
    complex_decoder.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[ResNet+ComplexDec] Epoch {epoch+1}/{EPOCHS}")
        for images_32, _ in pbar:
            images_32 = images_32.to(device)
            images_224 = resize_224(images_32)
            
            features = resnet(images_224)
            reconstructed_32 = complex_decoder(features)
            
            loss = criterion(reconstructed_32, images_32)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images_32.size(0)
            pbar.set_postfix({'loss': loss.item()})
        logging.info(f"[ResNet+ComplexDec] Epoch {epoch+1}, Loss: {running_loss/len(train_loader.dataset):.4f}")

    torch.save(resnet.state_dict(), os.path.join(SAVE_DIR, "resnet_encoder.pth"))
    torch.save(complex_decoder.state_dict(), os.path.join(SAVE_DIR, "complex_decoder.pth"))

if __name__ == "__main__":
    main()
