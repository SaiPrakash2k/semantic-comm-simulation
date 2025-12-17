import torch
import torch.nn as nn
import torchvision.models as models

# --- PRETRAINED MODELS (Transfer Learning) ---

class PretrainedMobileNetEncoder(nn.Module):
    """
    Local Encoder: Wraps MobileNetV3-Small (Pretrained on ImageNet).
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        # Load pretrained MobileNetV3 Small
        # We need to handle weights='DEFAULT' or similar depending on torchvision version
        # For compatibility, we'll try 'DEFAULT' first.
        try:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        except:
             # Fallback for older torchvision
             self.backbone = models.mobilenet_v3_small(pretrained=True)
             
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # The classifier of MobileNetV3 Small has a structure ending in Linear(576, 1000).
        # We replace it to map to our encoded_space_dim (512).
        # Backbone classifier: Sequential(Linear, Hardswish, Dropout, Linear)
        # We replace the final Linear layer.
        
        # Check input features of the last layer
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(num_features, encoded_space_dim)
        
        # We WANT to train this new linear layer, so ensure it requires grad
        self.backbone.classifier[-1].weight.requires_grad = True
        self.backbone.classifier[-1].bias.requires_grad = True

    def forward(self, x):
        # MobileNet expects normalized input, but we'll assume standard scaling happens outside or robust enough
        return self.backbone(x)

class PretrainedResNetEncoder(nn.Module):
    """
    Edge Encoder: Wraps ResNet-50 (Pretrained on ImageNet).
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights='DEFAULT')
        except:
            self.backbone = models.resnet50(pretrained=True)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # ResNet fc layer is Linear(2048, 1000). Replace it.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, encoded_space_dim)
        
        # Ensure new layer is trainable
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

# --- SIMPLE MODELS (Reuse Decoder for Local) ---

# --- SIMPLE MODELS (Reuse Decoder for Local) ---

class SimpleDecoder(nn.Module):
    """
    Lightweight Decoder for Local Processing.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 8, 8))
        self.decoder_conv = nn.Sequential(
            # Input: 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)   # -> 3 x 32 x 32
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# --- COMPLEX MODELS (Reuse Decoder for Edge) ---
# ComplexDecoder is also generic (takes 512 vector). So we can reuse it.

class ComplexDecoder(nn.Module):
    """
    Heavyweight Decoder for Edge/Cloud Processing.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))
        self.decoder_conv = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # -> 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # -> 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # -> 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1)                                # -> 3 x 32 x 32 (Refine)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# Keep legacy classes for now to avoid breaking imports immediately, but user wants replacement.
# I will NOT remove SimpleEncoder/ComplexEncoder definitions yet, just add the new ones above.

