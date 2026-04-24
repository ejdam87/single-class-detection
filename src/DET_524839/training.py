# STUDENT's UCO: 524839

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import albumentations as A
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph

from dataset import LabeledDetectionDataset, collate_labeled
from network import MyFRCNNDetector, WrapModel  # network is also standard package
from preprocessing import explore_dataset, train_val_test_split, MEAN, STD
from type_signature import ImageSample, BBoxesDict
from callbacks import EarlyStopping, BestModelLogger, Callback


# --- Training and Validation part
TRAIN_CONFIG = {
    "epochs": 12,
    "batch_size": 2,
    "num_workers": 4,
    "optimizer": torch.optim.AdamW,
    "optimizer_params": {"lr": 1e-4},
    "callbacks": {
        "early_stopping": EarlyStopping(patience=3),
        "best_model_logger": BestModelLogger(save_path="model.pt"),
    },
    "final_run": True,
}

TRAIN_TRANSFORMS = A.Compose(
    [
        # Geometric augmentations
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-5, 5),
            p=0.5,
        ),
        # Color augmentations
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        # Blur / noise
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=1.0,
        ),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],  # required by albumentations
        min_visibility=0.3,
    ),
)

VAL_TRANSFORMS = A.Compose(
    [
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=1.0,
        ),
    ]
)


def draw_network_architecture(net: MyFRCNNDetector) -> None:
    draw_graph(
        WrapModel(net),
        torch.rand(1, 3, 512, 1024),
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def loss_batch(
    model: MyFRCNNDetector,
    xb: list[ImageSample],
    yb: list[BBoxesDict],
    dev: torch.device,
    opt: Optimizer = None,
) -> float:

    xb = [x.to(dev) for x in xb]

    yb = [{k: v.to(dev) for k, v in t.items()} for t in yb]
    loss_dict = model(xb, yb)
    loss = sum(loss_dict.values())

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


def train_epoch(
    model: MyFRCNNDetector, train_dl: DataLoader, dev: torch.device, opt: Optimizer
) -> float:

    model.train()
    loss = 0
    for xb, yb, _ in tqdm(train_dl, total=len(train_dl), leave=False):
        b_loss = loss_batch(model, xb, yb, dev, opt)
        loss += b_loss

    return loss / len(train_dl)


def val_epoch(model: MyFRCNNDetector, val_dl: DataLoader, dev: torch.device) -> float:
    # Loss can be computed only in train mode for frcnn. Moreover, it is safe because the model uses
    # frozen batch norm layers, no dropouts etc.
    model.train()

    loss = 0
    with torch.no_grad():
        for xb, yb, _ in tqdm(val_dl, total=len(val_dl), leave=False):
            loss += loss_batch(model, xb, yb, dev)

    return loss / len(val_dl)


def fit(
    net: MyFRCNNDetector,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    callbacks: dict[str, Callback],
) -> tuple[list[float], list[float]]:

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(TRAIN_CONFIG["epochs"]):
        tl = train_epoch(net, train_dataloader, device, optimizer)
        vl = val_epoch(net, val_dataloader, device)
        train_losses.append(tl)
        val_losses.append(vl)

        callbacks["best_model_logger"].on_epoch_end(epoch, vl, net)
        should_stop = callbacks["early_stopping"].on_epoch_end(epoch, vl, net)
        if should_stop:
            break

    print("Training finished!")
    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing with {device}!")

    df = explore_dataset(dataset_path)
    print("Data frame prepared!")

    if TRAIN_CONFIG["final_run"]:
        train_df, val_df = train_val_test_split(
            df, val_only=True
        )  # no need for test set on the final run
    else:
        train_df, val_df, _ = train_val_test_split(
            df
        )  # since its seeded, test set can be used elsewhere for testing

    print("Data splitted!")

    train_dataset = LabeledDetectionDataset(train_df, TRAIN_TRANSFORMS)
    val_dataset = LabeledDetectionDataset(val_df, VAL_TRANSFORMS)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
        shuffle=True,
        collate_fn=collate_labeled,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
        collate_fn=collate_labeled,
    )

    print("Dataloaders created!")

    net = MyFRCNNDetector()
    net = net.to(device)
    draw_network_architecture(net)
    print("Network created!")

    optimizer = TRAIN_CONFIG["optimizer"](
        net.parameters(), **TRAIN_CONFIG["optimizer_params"]
    )
    print("Training started!")

    train_losses, val_losses = fit(
        net,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        TRAIN_CONFIG["callbacks"],
    )
    print("Training finished!")

    plot_learning_curves(train_losses, val_losses)
    print("Learning curves saved!")


# ---

# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
