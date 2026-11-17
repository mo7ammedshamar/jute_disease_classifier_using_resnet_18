"""Utility functions for training, reproducibility, plotting, and checkpointing."""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
) -> None:
    """Plots training and validation loss & accuracy curves side-by-side using Matplotlib.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: File path where the plot figure will be saved (e.g. .png).
        show_plot: Whether to display the plot interactively.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # --- Subplot 1: Loss curves ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", linewidth=2, markersize=5, label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r--s", linewidth=2, markersize=5, label="Val Loss")
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Annotate min val loss
    min_val_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    min_val_loss = min(history["val_loss"])
    plt.scatter([min_val_loss_epoch], [min_val_loss], color="red", s=100, zorder=5)
    plt.annotate(
        f"Min: {min_val_loss:.4f}\n(Epoch {min_val_loss_epoch})",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(min_val_loss_epoch, min_val_loss + 0.05),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
    )

    # --- Subplot 2: Accuracy curves ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", linewidth=2, markersize=5, label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], "g--^", linewidth=2, markersize=5, label="Val Accuracy")
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Annotate max val accuracy
    max_val_acc_epoch = int(np.argmax(history["val_acc"])) + 1
    max_val_acc = max(history["val_acc"])
    plt.scatter([max_val_acc_epoch], [max_val_acc], color="green", s=100, zorder=5)
    plt.annotate(
        f"Best: {max_val_acc:.2f}%\n(Epoch {max_val_acc_epoch})",
        xy=(max_val_acc_epoch, max_val_acc),
        xytext=(max_val_acc_epoch, max_val_acc - 5.0 if max_val_acc > 50 else max_val_acc + 5.0),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss and accuracy curves saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
) -> None:
    """Plots and saves a normalized confusion matrix heatmap using Matplotlib."""
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    # Row-normalized matrix
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix.astype("float"), row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(8, 7))
    plt.imshow(norm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Label every cell with percentage and count
    thresh = norm_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            cell_text = f"{norm_matrix[i, j]:.1%}\n({matrix[i, j]})"
            color = "white" if norm_matrix[i, j] > thresh else "black"
            plt.text(j, i, cell_text, horizontalalignment="center", verticalalignment="center", color=color, fontsize=9)

    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    class_names: List[str],
    filepath: Union[str, Path],
) -> None:
    """Saves model weights and training state checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Loads model weights and returns the checkpoint dictionary."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_history(history: Dict[str, List[float]], filepath: Union[str, Path]) -> None:
    """Saves training history metrics dictionary to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
