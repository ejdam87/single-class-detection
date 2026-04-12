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
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph


from dataset import LabeledDetectionDataset, collate_labeled
from network import FRCNNDetector
from preprocessing import explore_dataset, train_val_test_split, MEAN, STD


# --- Training and Validation part
TRAIN_CONFIG = {
    "epochs": 3,
    "batch_size": 1,
    "optimizer": torch.optim.AdamW,
    "optimizer_params": {
        "lr": 1e-3
    },
}

TRAIN_TRANSFORMS =A.Compose([
        # Geometric augmentations
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5
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
        label_fields=["labels"], # required by albumentations
        min_visibility=0.3
    )
)

VAL_TRANSFORMS = A.Compose([
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=1.0,
        ),
    ]
)

INFERENCE_TRANSFORMS = A.Compose([
        A.Normalize(
            mean=MEAN,
            std=STD,
            max_pixel_value=1.0,
        ),
    ]
)


# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


def plot_learning_curves(train_losses: list[float], validation_losses: list[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def loss_batch(
          model: nn.Module,
          xb: Tensor,
          yb: Tensor,
          dev: torch.device,
          opt: Optimizer=None
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
          model: nn.Module,
          train_dl: DataLoader,
          dev: torch.device,
          opt: Optimizer
    ) -> float:

        model.train()
        loss = 0
        for xb, yb, _ in tqdm(train_dl, total=len(train_dl), leave=False):
            b_loss = loss_batch(model, xb, yb, dev, opt)
            loss += b_loss

        return loss / len(train_dl)


def val_epoch(model: nn.Module, val_dl: DataLoader, dev: torch.device) -> float:
        model.eval()

        loss = 0
        with torch.no_grad():
            for xb, yb, _ in tqdm(val_dl, total=len(val_dl), leave=False):
                loss += loss_batch(model, xb, yb, dev)

        return  loss/ len(val_dl)


def fit(
    net: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[list[float], list[float]]:

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(TRAIN_CONFIG["epochs"]):
        tl = train_epoch(net, train_dataloader, device, optimizer)
        vl = val_epoch(net, val_dataloader, device)

        train_losses.append(tl)
        val_losses.append(vl)

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

    train_df, val_df, _ = train_val_test_split(df)
    print("Data splitted!")

    train_dataset = LabeledDetectionDataset(train_df, TRAIN_TRANSFORMS)
    val_dataset = LabeledDetectionDataset(val_df, VAL_TRANSFORMS)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_labeled)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"], collate_fn=collate_labeled)
    print("Dataloaders created!")

    net = FRCNNDetector()
    net = net.to(device)
    # input_sample = torch.zeros((3, 512, 1024))
    # draw_network_architecture(net, input_sample)
    print("Network created!")

    optimizer = TRAIN_CONFIG["optimizer"](net.parameters(), **TRAIN_CONFIG["optimizer_params"])
    print("Training started!")

    train_losses, val_losses = fit(
        net,
        train_dataloader,
        val_dataloader,
        optimizer,
        device
    )
    print("Training finished!")

    torch.save(net.state_dict(), "model.pt")
    print("Model saved!")

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
