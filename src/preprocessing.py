"""This file contains exploration and preprocessing of the dataset."""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import Tensor
from pathlib import Path
from dataset import DetectionDataset


TEST_SIZE = 1 / 10  # (10% from the whole)
VAL_SIZE = 1 / 9  # (10% from the whole)
RANDOM_STATE = 42

# pre-computed for training data
MEAN = [81.7474, 59.7219, 70.0178]
STD = [47.6729, 41.8091, 46.3294]


def train_val_test_split(
    df: pd.DataFrame,
    val_only: bool = False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]
):
    trv_df, te_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["city"]
    )

    # no need for separate test set
    if val_only:
        return trv_df, te_df

    tr_df, v_df = train_test_split(
        trv_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=trv_df["city"]
    )

    return tr_df, v_df, te_df


def explore_dataset(dataset_path: Path) -> pd.DataFrame:
    bbox_path = dataset_path / "bbox"
    images_path = dataset_path / "img"

    images = images_path.rglob("*.png")
    bboxes = bbox_path.rglob("*.csv")

    data = {
        "image_path": [],
        "bbox_path": [],
        "city": [],
        "n_objects": [],
    }

    for im_path, bbox_path in zip(images, bboxes):
        data["image_path"].append(str(im_path))
        data["bbox_path"].append(str(bbox_path))
        data["city"].append(im_path.parent.name)
        data["n_objects"].append(len(pd.read_csv(str(bbox_path))))

    final_df = pd.DataFrame(data)
    return final_df


# This was used to compute mean and standard deviation per color channel used in normalization
def compute_mean_std(dataset: DetectionDataset) -> tuple[Tensor, Tensor]:
    sm = torch.zeros(3, dtype=torch.float64)
    sq_sm = torch.zeros(3, dtype=torch.float64)
    total = 0

    for i in range(len(dataset)):
        im, _ = dataset[i]
        im = im.float()
        sm += torch.sum(im, dim=[1, 2])
        sq_sm += torch.sum(im**2, dim=[1, 2])
        total += im.shape[1] * im.shape[2]

    mean = sm / total
    var = (sq_sm / total) - (mean**2)
    std = torch.sqrt(var)
    return mean, std
