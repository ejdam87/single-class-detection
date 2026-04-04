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

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        assert "image_path" in df.columns, "Need a path to the image"

        self.df = df
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.df)

    def _get_image(self, sample: pd.Series) -> Tensor:
        img = Image.open(sample["image_path"])
        img = np.array(img)
        img = self.to_tensor(image=img)["image"]
        return img

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        pass

class LabeledDetectionDataset(DetectionDataset[LabeledSample]):

    def __init__(self, df: pd.DataFrame) -> None:
        assert "bbox_path" in df.columns, "Need labels for labeled dataset"
        super().__init__(df=df)

    def __getitem__(self, idx: int) -> LabeledSample:
        sample = self.df.iloc[idx]
        img = self._get_image(sample)

        bbox_df = pd.read_csv(sample["bbox_path"])
        bbox_torch = torch.tensor(bbox_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)

        metadata = {
            "filename": Path(sample["image_path"]).name,
            "n_labeled_cars": sample["n_objects"],
            "city": sample["city"],
        }

        return img, metadata, bbox_torch


class UnlabeledDetectionDataset(DetectionDataset[UnlabeledSample]):

    def __getitem__(self, idx: int) -> UnlabeledSample:
        sample = self.df.iloc[idx]
        img = self._get_image(sample)
        metadata = {
            "filename": Path(sample["image_path"]).name,
            "city": sample["city"],
        }

        return img, metadata
