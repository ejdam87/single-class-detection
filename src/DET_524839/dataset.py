# STUDENT's UCO: 524839

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic, Any

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from albumentations import ToTensorV2

ImageSample = Tensor  # (3, H, W)
Bboxes = Tensor # (N, 4)

Metadata = dict[str, Any]

LabeledSample = tuple[ImageSample, Metadata, Bboxes]
UnlabeledSample = tuple[ImageSample, Metadata]

T = TypeVar("T")

class DetectionDataset(ABC, Dataset[T], Generic[T]):

    def __init__(self, image_paths: list[Path]) -> None:
        super().__init__()

        self.image_paths = image_paths
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_image(self, idx: int) -> Tensor:
        img = Image.open(str(self.image_paths[idx]))
        img = np.array(img)
        img = self.to_tensor(image=img)["image"]
        return img

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        pass

class LabeledDetectionDataset(DetectionDataset[LabeledSample]):

    def __init__(
            self,
            image_paths: list[Path],
            bbox_paths: list[Path],
        ) -> None:

        assert len(image_paths) == len(bbox_paths), "Expecting same amount of images and labels"
        super().__init__(image_paths=image_paths)

        self.bbox_paths = bbox_paths

    def __getitem__(self, idx: int) -> LabeledSample:
        img = self._get_image(idx)

        bbox_df = pd.read_csv(str(self.bbox_paths[idx]))
        bbox_torch = torch.tensor(bbox_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        metadata = {
            "filename": self.image_paths[idx].name,
            "n_labeled_cars": len(bbox_df),
        }

        return img, metadata, bbox_torch


class UnlabeledDetectionDataset(DetectionDataset[UnlabeledSample]):

    def __getitem__(self, idx: int) -> UnlabeledSample:
        img = self._get_image(idx)
        metadata = {
            "filename": self.image_paths[idx].name,
        }

        return img, metadata
