# STUDENT's UCO: 524839

"""This file contains type aliases for objects used around the whole project."""

from typing import Any

from torch import Tensor


ImageSample = Tensor  # (3, H, W)
BboxesTensor = Tensor  # (N, 4)
BBoxesDict = dict[str, Tensor]  # {boxes: BboxesTensor, labels: "ones"}

Metadata = dict[str, Any]

LabeledSample = tuple[ImageSample, BboxesTensor, Metadata]
UnlabeledSample = tuple[ImageSample, Metadata]

InferenceOutput = list[dict[str, Tensor]]  # boxes, labels, scores
TrainingOutput = dict[
    str, Tensor
]  # regression and classification losses (final detection and region proposal)
