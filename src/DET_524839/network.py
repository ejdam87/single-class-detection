# STUDENT's UCO: 524839

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
import torch.nn as nn

class GridDetector(nn.Module):
    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Output: [batch, grid_size*grid_size, 5] → [x,y,w,h,obj_score]
        self.pred_head = nn.Linear(64, 5)

    def forward(self, x):
        features = self.conv(x)
        features = features.flatten(2).transpose(1, 2)  # [B, S*S, C]
        preds = self.pred_head(features)                # [B, S*S, 5]
        preds[..., 4] = torch.sigmoid(preds[..., 4])   # objectness score
        return preds
