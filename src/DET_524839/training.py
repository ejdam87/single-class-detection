# STUDENT's UCO: 524839

# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import pandas as pd
from tqdm import tqdm
from dataset import LabeledDetectionDataset, UnlabeledDetectionDataset, DetectionDataset
from network import SimpleGridDetector
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchview import draw_graph
from sklearn.model_selection import train_test_split


# --- Exploration and Pre-processing part
VAL_SIZE = 0.2
RANDOM_STATE = 42

# pre-computed for training data
MEAN = [81.7474, 59.7219, 70.0178]
STD = [47.6729, 41.8091, 46.3294]


def train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_df, v_df = train_test_split(df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df["city"])
    return t_df, v_df

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
        data["image_path"].append( str(im_path) )
        data["bbox_path"].append( str(bbox_path) )
        data["city"].append( im_path.parent.name )
        data["n_objects"].append( len(pd.read_csv( str(bbox_path) )) )

    final_df = pd.DataFrame(data)
    return final_df

def compute_mean_std(dataset: DetectionDataset) -> tuple[Tensor, Tensor]:
    sm = torch.zeros(3, dtype=torch.float64)
    sq_sm = torch.zeros(3, dtype=torch.float64)
    total = 0

    for i in range(len(dataset)):
        im, _ = dataset[i]
        im = im.float()
        sm += torch.sum(im, dim=[1, 2])
        sq_sm += torch.sum(im ** 2, dim=[1, 2])
        total += im.shape[1] * im.shape[2]

    mean = sm / total
    var = (sq_sm / total) - (mean ** 2)
    std = torch.sqrt(var)
    return mean, std
# ---


# --- Training and Validation part
TRAIN_CONFIG = {
    "use_precomputed_mean_std": True,
    "epochs": 3,
    "batch_size": 64,
    "optimizer": torch.optim.AdamW,
    "optimizer_params": {
        "lr": 1e-3
    },
    "loss": torch.nn.MSELoss,
    "loss_params": {},
}


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
          loss_func: nn.Module,
          xb: Tensor,
          yb: Tensor,
          dev: torch.device,
          opt: Optimizer=None
    ) -> float:

    xb, yb = xb.to(dev), yb.to(dev)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def train_epoch(
          model: nn.Module,
          train_dl: DataLoader,
          loss_func: nn.Module,
          dev: torch.device,
          opt: Optimizer
    ) -> float:

        model.train()
        loss = 0
        for xb, yb in tqdm(train_dl, total=len(train_dl), leave=False):
            b_loss = loss_batch(model, loss_func, xb, yb, dev, opt)
            loss += b_loss
            size += len(xb)

        return loss


def val_epoch(model: nn.Module, val_dl: DataLoader, loss_func: nn.Module, dev: torch.device) -> float:
        model.eval()

        loss = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, total=len(val_dl), leave=False):
                loss += loss_batch(model, loss_func, xb, yb, dev)
            
        return loss


def fit(
    net: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[list[float], list[float]]:

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(TRAIN_CONFIG["epochs"]):
        tl = train_epoch(net, train_dataloader, loss, device, optimizer)
        vl = val_epoch(net, val_dataloader, loss, device)

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

    train_df, val_df = train_val_split(df)
    print("Data splitted!")

    train_dataset = LabeledDetectionDataset(train_df)
    val_dataset = LabeledDetectionDataset(val_df)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG["batch_size"])
    print("Dataloaders created!")

    net = SimpleGridDetector(grid_size=10)
    input_sample = torch.zeros((1, 512, 1024))
    draw_network_architecture(net, input_sample)

    optimizer = TRAIN_CONFIG["optimizer"](net.parameters(), **TRAIN_CONFIG["optimizer_params"])
    loss = TRAIN_CONFIG["loss"](**TRAIN_CONFIG["loss_params"])

    print("Training started!")
    train_losses, val_losses = fit(
        net,
        train_dataloader,
        val_dataloader,
        loss,
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
