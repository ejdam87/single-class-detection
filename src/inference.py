# STUDENT's UCO: 524839

# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import torch
import pandas as pd
import albumentations as A
from tqdm import tqdm

from network import MyFRCNNDetector
from dataset import UnlabeledDetectionDataset, collate_unlabeled
from preprocessing import MEAN, STD, explore_dataset, train_val_test_split
from type_signature import ImageSample, Metadata


INFERENCE_CONFIG = {
    "arbitrary_samples": True,  # whether to use test set on inference on new samples
    "confidence_t": 0.0,  # do not filter any bboxes after non maxima suppresion since the evaluation metric is AP
    "batch_size": 2,
    "num_workers": 2,
}

INFERENCE_TRANSFORMS = A.Compose(
    [
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=1.0,
        ),
    ]
)


def inference_batch(
    model: MyFRCNNDetector, batch: list[ImageSample], metadata: list[Metadata]
) -> list[pd.DataFrame]:
    outputs = model(batch)
    dfs = []

    for output, meta in zip(outputs, metadata):
        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()

        labels = output["labels"].cpu()
        assert torch.all(labels > 0), (
            f"Background label detected in predictions: {labels}"
        )

        keep = scores >= INFERENCE_CONFIG["confidence_t"]
        boxes = boxes[keep]
        scores = scores[keep]

        filename = meta["filename"]

        if len(boxes) == 0:
            df = pd.DataFrame(
                columns=["filename", "xmin", "xmax", "ymin", "ymax", "confidence"]
            )
            dfs.append(df)
            continue

        xmin = boxes[:, 0].numpy()
        ymin = boxes[:, 1].numpy()
        xmax = boxes[:, 2].numpy()
        ymax = boxes[:, 3].numpy()
        confidence = scores.numpy()

        df = pd.DataFrame(
            {
                "filename": [filename] * len(xmin),
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "confidence": confidence,
            }
        )

        dfs.append(df)

    return dfs


# declaration for this function should not be changed
@torch.no_grad()  # do not calculate the gradients
def inference(dataset_path: Path, model_path: Path) -> None:
    """Performs inference on the given dataset using the specified model.

    Args:
        dataset_path: Path to the dataset. The function processes all PNG images in
            this directory (optionally recursively in its subdirectories).
        model_path: Path to the model file.

    Saves:
        predictions to 'output_predictions' folder. The files can be saved in a flat
            structure with the same name as the input file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing with {device}!")

    model = MyFRCNNDetector()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print("Model loaded!")

    if INFERENCE_CONFIG["arbitrary_samples"]:
        df = pd.DataFrame({"image_path": list(dataset_path.rglob("*.png"))})
    else:  # testing purposes
        df = explore_dataset(dataset_path)
        _, _, df = train_val_test_split(df, val_only=False)

    ds = UnlabeledDetectionDataset(df, INFERENCE_TRANSFORMS)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=INFERENCE_CONFIG["batch_size"],
        num_workers=INFERENCE_CONFIG["num_workers"],
        collate_fn=collate_unlabeled,
    )
    print("Dataloader created!")

    out_dir = Path("output_predictions")
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Inference started!")
    for xb, metadata in tqdm(dl, total=len(dl), leave=False):
        xb = [x.to(device) for x in xb]
        dfs = inference_batch(model, xb, metadata)
        for i, df in enumerate(dfs):
            df.to_csv(str((out_dir / metadata[i]["filename"]).with_suffix(".csv")))

    print("Inference finished!")


# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
