import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) Layer.
    Learns which channels contain key texture info and re-weights them.
    """
    def __init__(self, n_feats, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB).
    Basic building block: Conv -> ReLU -> Conv -> CA -> Residual
    """
    def __init__(self, n_feats, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            ChannelAttention(n_feats, reduction)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x

class ResidualGroup(nn.Module):
    """
    Residual Group (RG).
    A group of RCABs with a skip connection at the end.
    """
    def __init__(self, n_feats, n_resblocks, reduction):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feats, reduction) for _ in range(n_resblocks)
        ]
        modules_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res + x

class SuperResolutionModel(nn.Module):
    """
    RCAN Architecture (Optimized for <5M params).
    Configuration:
      - Channels: 64 (standard for SR)
      - Groups: 5
      - Blocks per Group: 11
      - Total Depth: ~55 non-linear layers
      - Upsampling: Progressive (2x -> 2x) for efficiency
    """
    def __init__(self):
        super(SuperResolutionModel, self).__init__()
        
        # --- Configuration ---
        n_feats = 64
        n_resgroups = 5
        n_resblocks = 11
        reduction = 16
        scale = 4 
        
        # 1. Shallow feature extraction
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        
        # 2. Deep feature extraction (Residual Groups)
        modules_body = [
            ResidualGroup(n_feats, n_resblocks, reduction) for _ in range(n_resgroups)
        ]
        modules_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*modules_body)
        
        # 3. Upsampling (Progressive: 2x then 2x = 4x)
        # Progressive is more parameter efficient than direct 4x
        self.tail = nn.Sequential(
            # 2x
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            # 2x
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            # Final reconstruction
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.head(x)
        
        res = self.body(x)
        res += x  # Global Long Skip Connection
        
        out = self.tail(res)
        # out: (B, 3, 128, 128)
        return out