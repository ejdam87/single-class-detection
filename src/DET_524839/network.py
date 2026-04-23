# STUDENT's UCO: 524839

# Description:
# This file should contain network class. The class should subclass the torch.nn.Module class.

import torch
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.image_list import ImageList

from type_signature import ImageSample, BBoxesDict, TrainingOutput, InferenceOutput


class MyFRCNNDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # Backbone -> extracting features (pyramid enables multi-scale features)
        # -------------------------
        backbone_base = resnet50(weights="DEFAULT")

        self.backbone = BackboneWithFPN(
            backbone_base,
            return_layers={
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3",
            },
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )

        # -------------------------
        # RPN -> proposing regions with objects
        # -------------------------
        anchor_generator = AnchorGenerator(
            sizes=(
                (32,),
                (64,),
                (128,),
                (256,),
                (512,),
            ),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        rpn_head = RPNHead(
            in_channels=256,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000),
            nms_thresh=0.7,
        )

        # -------------------------
        # RoI pooling
        # -------------------------
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        # -------------------------
        # Box head + predictor
        # -------------------------
        representation_size = 1024

        # dense representation network
        box_head = TwoMLPHead(
            in_channels=256 * 7 * 7,
            representation_size=representation_size,
        )

        # classification, bbox regression
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes=2,  # background, car
        )

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

    def forward(
        self, images: list[ImageSample], bboxes: list[BBoxesDict] | None = None
    ) -> TrainingOutput | InferenceOutput:

        # prepare images
        image_sizes = [img.shape[-2:] for img in images]
        images_tensor = torch.stack(images)  # Tensor (B, C, H, W)

        # get feature maps
        features = self.backbone(
            images_tensor
        )  # dict: layer name -> Tensor (B, 256, H, W)

        # to be able to use rpn (enables working with padded tensors)
        images_list = ImageList(images_tensor, image_sizes)

        # region proposal
        proposals, rpn_losses = self.rpn(
            images_list, features, bboxes
        )  # list of tensors (K, 4), objectness and bbox regression loss
        detections, detector_losses = self.roi_heads(
            features, proposals, image_sizes, bboxes
        )  # same structure as above

        # continue structure of submodules returning losses on training only
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses

        return detections


# Visualization purposes
class WrapModel(torch.nn.Module):
    def __init__(self, model: MyFRCNNDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> InferenceOutput:
        # x: (B, C, H, W)
        images = [img for img in x]
        return self.model(images)
