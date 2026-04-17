# STUDENT's UCO: 524839

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Generic

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from albumentations import ToTensorV2
from albumentations.core.composition import TransformType

from type_signature import (
    ImageSample,
    BboxesTensor,
    LabeledSample,
    UnlabeledSample,
    Metadata,
    BBoxesDict,
)


T = TypeVar("T")


class DetectionDataset(ABC, Dataset[T], Generic[T]):
    def __init__(
        self, df: pd.DataFrame, transforms: TransformType | None = None
    ) -> None:
        super().__init__()
        assert "image_path" in df.columns, "Need a path to the image"

        self.df = df
        self.to_tensor = ToTensorV2()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transforms(
        self, image: Image, bboxes: BboxesTensor | None = None
    ) -> ImageSample | tuple[ImageSample, BboxesTensor]:
        if self.transforms is not None:
            if bboxes is None:
                image = self.transforms(image=image)["image"]
            else:
                transformed = self.transforms(
                    image=image,
                    bboxes=bboxes.numpy().tolist(),
                    labels=[1] * len(bboxes),
                )
                image = transformed["image"]
                bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                if len(bboxes) == 0:
                    bboxes = torch.zeros((0, 4), dtype=torch.float32)

        image = self.to_tensor(image=image)["image"]

        if bboxes is None:
            return image

        return image, bboxes

    def _get_image(self, sample: pd.Series) -> Image:
        img = Image.open(sample["image_path"])
        img = np.array(img)
        return img

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        pass


class LabeledDetectionDataset(DetectionDataset[LabeledSample]):
    def __init__(
        self, df: pd.DataFrame, transforms: TransformType | None = None
    ) -> None:
        assert "bbox_path" in df.columns, "Need labels for labeled dataset"
        super().__init__(df=df, transforms=transforms)

    def __getitem__(self, idx: int) -> LabeledSample:
        sample = self.df.iloc[idx]
        img = self._get_image(sample)
        bbox_df = pd.read_csv(sample["bbox_path"])
        bbox_torch = torch.tensor(
            bbox_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
        )
        img, bbox_torch = self._apply_transforms(img, bbox_torch)

        metadata = {
            "filename": Path(sample["image_path"]).name,
            "n_labeled_cars": sample["n_objects"],
            "city": sample["city"],
        }

        return img, bbox_torch, metadata


class UnlabeledDetectionDataset(DetectionDataset[UnlabeledSample]):
    def __getitem__(self, idx: int) -> UnlabeledSample:
        sample = self.df.iloc[idx]
        img = self._get_image(sample)
        img = self._apply_transforms(img)

        if "city" in self.df.columns:
            metadata = {
                "filename": Path(sample["image_path"]).name,
                "city": sample["city"],
            }
        else:
            metadata = {
                "filename": Path(sample["image_path"]).name,
            }

        return img, metadata


def collate_unlabeled(
    batch: list[UnlabeledSample],
) -> tuple[list[ImageSample], list[Metadata]]:
    images = []
    metas = []
    for img, meta in batch:
        images.append(img)
        metas.append(meta)
    return images, metas


def collate_labeled(
    batch: list[LabeledSample],
) -> tuple[list[ImageSample], list[BBoxesDict], list[Metadata]]:
    images = []
    metas = []
    bboxes_list = []
    for img, bboxes, meta in batch:
        images.append(img)
        metas.append(meta)
        bboxes_list.append(
            {
                "boxes": bboxes,
                "labels": torch.ones((bboxes.shape[0],), dtype=torch.int64),
            }
        )

    return images, bboxes_list, metas
