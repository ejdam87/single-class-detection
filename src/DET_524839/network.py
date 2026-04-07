# STUDENT's UCO: 524839

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
import torch.nn as nn
from torch import Tensor


"""
Problems:
- cannot predict more than grid_size^2 objects
- if object is on the border of cells (or spans multiple cells), it is missed
- always predicts grid_size^2 objects -> no prunning
- bbox non-normalized
"""
class SimpleGridDetector(nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )

        # Each grid cell predicts [x, y, w, h, obj_score]
        self.pred_head = nn.Linear(64, 5)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv(x) # [B, C, grid_size, grid_size]
        B, C, H, W = features.shape

        features = features.view(B, C, H * W) # [B, C, grid_size * grid_size]
        features = features.permute(0, 2, 1) # [B, grid_size * grid_size, C]

        preds = self.pred_head(features) # [B, grid_size * grid_size, 5]
        preds[..., 4] = torch.sigmoid(preds[..., 4]) # [B, grid_size * grid_size, 5]

        return preds
