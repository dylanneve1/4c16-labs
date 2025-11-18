import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out


class SuperResolutionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionModel, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        
        # Residual blocks for feature learning
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(10)]
        )
        
        # Middle convolution
        self.conv_mid = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Upsampling via pixel shuffle (4x = 2x → 2x)
        self.upscale = nn.Sequential(
            # First 2x upsampling
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            
            # Second 2x upsampling
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            
            # Final reconstruction
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.conv_first(x)          # (B, 64, 32, 32)
        
        residual = x
        x = self.residual_blocks(x)     # (B, 64, 32, 32)
        x = self.conv_mid(x)            # (B, 64, 32, 32)
        x = x + residual                # Skip connection
        
        x = self.upscale(x)             # (B, 3, 128, 128)
        
        return x