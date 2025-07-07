'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

NUM_CLASSES = 4

    

# -----------------------------
# 3D UNet with Self Attention
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.attention = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        x = x + self.ff(self.norm2(x))
        return x.transpose(1, 2).view(B, C, D, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.residual = residual
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch)
        )

    def forward(self, x):
        out = self.block(x)
        return F.gelu(out + x) if self.residual else out


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, in_ch // 2)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()

        # downsampling
        self.input = DoubleConv(in_channels, 16) # → (16, 128, 128, 128)

        self.down1 = Down(16, 32) # → (32,  64,  64,  64)
        self.down2 = Down(32, 64) # → (64,  32,  32,  32)
        self.down3 = Down(64, 64) # → (64,  16,  16,  16)

        # bottleneck
        self.bot = nn.Sequential(
            DoubleConv(64, 128),  # → (128, 16, 16, 16)
            DoubleConv(128, 128), # → (128, 16, 16, 16)
            DoubleConv(128, 64)   # → (64,  16, 16, 16)
        )
        self.sa_mid = SelfAttention(64) # → (64,  16,  16,  16)

        # upsampling
        self.up1 = Up(128, 32) # → (32,  32,  32,  32)
        self.up2 = Up(64, 16) # → (16,  64,  64,  64)
        self.up3 = Up(32, 16) # → (16, 128, 128, 128)

        self.output = nn.Conv3d(16, num_classes, kernel_size=1) # → (4, 128, 128, 128)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_mid = self.bot(x4)
        x_mid = self.sa_mid(x_mid)

        x = self.up1(x_mid, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.output(x)
    

# -----------------------------
# Utility functions
# -----------------------------
def segment(img, model, device='cpu', raw=False):
    '''
    Uses the model to segment a 3D image

    Args:
        img (torch.Tensor or np.ndarray): of shape (3, 128, 128, 128).
        model (torch.nn.Module): the u-net model
        device (str): Device to run the model on 
        raw (bool): If True, returns raw model output of shape (4, 128, 128, 128).
                    If False, returns a decoded class map of shape (128, 128, 128).

    Returns:
        np.ndarray: Either the raw output of class scores or a class index map.
    '''
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    with torch.no_grad():
        img = img.unsqueeze(0).to(device)  # (C, D, H, W) -> (1, C, D, H, W)
        pred = model(img)
        if not raw:
            pred = torch.argmax(pred, dim=1)  
        pred = pred.cpu().squeeze().numpy()
        return pred
    



def save_checkpoint(state, filepath):
    print("✅ Saving checkpoint...\n")
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=False):
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)