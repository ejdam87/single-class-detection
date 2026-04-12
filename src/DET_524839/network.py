# STUDENT's UCO: 524839

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from type_signature import ImageSample, BBoxesDict, TrainingOutput, InferenceOutput


class FRCNNDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            min_size=512,
            max_size=1024,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
        )

        # Replace the head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    def forward(self, images: list[ImageSample], bboxes: list[BBoxesDict] | None=None) -> TrainingOutput | InferenceOutput:
        return self.model(images, bboxes)
