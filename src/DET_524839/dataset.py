# STUDENT's UCO: 524839

from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from albumentations import ToTensorV2

Image = Tensor  # (3, H, W)
Bboxes = Tensor # (N, 4)

Metadata = dict[str, str]

LabeledSample = tuple[Image, Metadata, Bboxes]
UnlabeledSample = tuple[Image, Metadata]


class LabeledDetectionDataset(Dataset[LabeledSample]):

    def __init__(
            self,
            image_paths: list[Path],
            bbox_paths: list[Path],
        ) -> None:

        assert len(image_paths) == len(bbox_paths), "Expecting same amount of images and labels"
        super().__init__()

        self.image_paths = image_paths
        self.bbox_paths = bbox_paths
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> LabeledSample:
        img = Image.open(str(self.image_paths[idx]))
        img = np.array(img)
        img = self.to_tensor(image=img)["image"]

        bbox_df = pd.read_csv(str(self.bbox_paths[idx]))
        bbox_torch = torch.tensor(bbox_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        metadata = {
            "filename": self.image_paths[idx].name,
            "n_labeled_cars": len(bbox_df),
        }

        return img, metadata, bbox_torch
