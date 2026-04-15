# models/fusion_frontend.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNAFMto8x8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(640, 128, 3, padding=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 32,  3, padding=1, bias=False), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32,  1,   3, padding=1, bias=True)
        )

    def forward(self, fm):  # [B,41,640]
        x = fm.permute(0,2,1)             # [B,640,41]
        x = self.conv(x)                  # [B,1,41]
        x = F.adaptive_avg_pool1d(x, 64)  # [B,1,64]
        x = x.view(x.size(0), 1, 8, 8)    # [B,1,8,8]
        
        x = x - x.amin(dim=(2,3), keepdim=True)
        denom = x.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        x = x / denom
        return x

class CGRConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1, bias=True)
        )
    def forward(self, cgr):
        x = self.conv(cgr)                # [B,1,8,8]
        x = x - x.amin(dim=(2,3), keepdim=True)
        denom = x.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        x = x / denom
        return x

class FusionToCapsInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fm8x8  = RNAFMto8x8()
        self.cgr8x8 = CGRConvStem()
        self.mix    = nn.Conv2d(2, 1, kernel_size=1, bias=True)

    def forward(self, fm_emb, cgr_img):
        f1 = self.fm8x8(fm_emb)                 # [B,1,8,8]
        f2 = self.cgr8x8(cgr_img)               # [B,1,8,8]
        x  = torch.cat([f1, f2], dim=1)         # [B,2,8,8]
        x  = self.mix(x)                        # [B,1,8,8]
        # 0..1
        #x = x - x.amin(dim=(2,3), keepdim=True)
        #denom = x.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)
        #x = x / denom
        return x
