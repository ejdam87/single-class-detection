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
from albumentations.core.composition import TransformType

ImageSample = Tensor  # (3, H, W)
Bboxes = Tensor # (N, 4)

Metadata = dict[str, Any]

LabeledSample = tuple[ImageSample, Metadata, Bboxes]
UnlabeledSample = tuple[ImageSample, Metadata]

T = TypeVar("T")

class DetectionDataset(ABC, Dataset[T], Generic[T]):

    def __init__(self, df: pd.DataFrame, transforms: TransformType|None=None) -> None:
        super().__init__()
        assert "image_path" in df.columns, "Need a path to the image"

        self.df = df
        self.to_tensor = ToTensorV2()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transforms(self, image: Image, bboxes: Bboxes | None=None) -> ImageSample | tuple[ImageSample, Bboxes]:
        if self.transforms is not None:
            if bboxes is None:
                image = self.transforms(image=image)["image"]
            else:
                transformed = self.transforms(image=image, bboxes=bboxes.numpy().tolist(), labels = [0] * len(bboxes))
                image = transformed["image"]
                image = self.to_tensor(image=image)["image"]
                bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                return image, bboxes

        image = self.to_tensor(image=image)["image"]
        return image

    def _get_image(self, sample: pd.Series) -> Image:
        img = Image.open(sample["image_path"])
        img = np.array(img)
        return img

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        pass

class LabeledDetectionDataset(DetectionDataset[LabeledSample]):

    def __init__(self, df: pd.DataFrame, transforms: TransformType | None=None) -> None:
        assert "bbox_path" in df.columns, "Need labels for labeled dataset"
        super().__init__(df=df, transforms=transforms)

    def __getitem__(self, idx: int) -> LabeledSample:
        sample = self.df.iloc[idx]
        img = self._get_image(sample)
        bbox_df = pd.read_csv(sample["bbox_path"])
        bbox_torch = torch.tensor(bbox_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
        img, bbox_torch = self._apply_transforms(img, bbox_torch)

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
        img = self._apply_transforms(img)
        metadata = {
            "filename": Path(sample["image_path"]).name,
            "city": sample["city"],
        }

        return img, metadata
