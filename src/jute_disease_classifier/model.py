"""ResNet-18 Transfer Learning Model for Jute Leaf Disease Classification."""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Builds and initializes a ResNet-18 model with a custom classification head.

    Args:
        num_classes: Number of target categories (default: 5).
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze_backbone: If True, freeze all convolutional layers in the backbone.
        dropout_rate: Dropout probability before the final linear classifier layer.

    Returns:
        Configured PyTorch nn.Module.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ResNet-18 final fully connected layer has 512 input features
    in_features = model.fc.in_features

    if dropout_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all parameters in the backbone for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns a tuple of (trainable_parameters, total_parameters)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
