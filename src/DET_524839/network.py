# STUDENT's UCO: 524839

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FRCNNDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Replace the head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)

    def forward(self, images: Tensor, bboxes: Tensor | None=None) -> dict | list:
        """
        images:  [B, C, H, W]
        targets: [B, N, 4]  (xyxy format assumed)

        Returns:
            During training: dict of losses
            During inference: list of dicts (boxes, labels, scores)
        """

        image_list = [img for img in images]
        if bboxes is not None:
            bbox_list = [
                {
                    "boxes": bboxes[i],  # [N, 4]
                    "labels": torch.ones(
                        (bboxes[i].shape[0],),
                        dtype=torch.int64,
                        device=bboxes.device,
                    )
                }
                for i in range(bboxes.shape[0])
            ]
        else:
            bbox_list = None

        return self.model(image_list, bbox_list)
