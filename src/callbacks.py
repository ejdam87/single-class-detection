# STUDENT's UCO: 524839

"""This file contains callbacks called on epoch end of each epoch."""

from abc import ABC, abstractmethod

from network import MyFRCNNDetector

import torch


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, val_loss: float, model: MyFRCNNDetector) -> bool:
        pass


class EarlyStopping(Callback):
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, epoch: int, val_loss: float, model: MyFRCNNDetector) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"[EarlyStopping] Early stopping triggered at epoch {epoch}")
            return True

        return False


class BestModelLogger(Callback):
    def __init__(self, save_path: str) -> None:
        self.best_loss = float("inf")
        self.best_model_state = None
        self.save_path = save_path

    def on_epoch_end(self, epoch: int, val_loss: float, model: MyFRCNNDetector) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)

            print(
                f"[BestModelLogger] New best model at epoch {epoch}, val_loss={val_loss:.6f}"
            )
            return True

        return False
