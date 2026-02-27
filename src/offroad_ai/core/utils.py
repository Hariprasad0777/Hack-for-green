"""
Utility modules for Offroad Terrain AI.

This module provides the core mathematical components including the implementation 
of the UltimateHybridLoss and performance metrics (mIoU) used for evaluation.
"""

from typing import Tuple, List

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from offroad_ai.core import config


class UltimateHybridLoss(nn.Module):
    """
    A weighted composite loss function for semantic segmentation.

    This loss combines Dice Loss, Focal Loss, and Weighted Cross-Entropy to 
    provide a robust objective function that handles high imbalanced classes 
    typical in offroad environments.

    Attributes:
        dice (smp.losses.DiceLoss): Multi-class Dice loss component.
        focal (smp.losses.FocalLoss): Focal loss for hard-example mining.
        ce (nn.CrossEntropyLoss): Class-weighted cross entropy.
    """

    def __init__(self, weights: torch.Tensor = config.CLASS_WEIGHTS):
        """
        Initializes the Hybrid Loss module.

        Args:
            weights (torch.Tensor): Class weights for Cross Entropy component.
        """
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the final composite loss value.

        Args:
            y_pred (torch.Tensor): Predicted logits of shape (N, C, H, W).
            y_true (torch.Tensor): Ground truth labels of shape (N, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return (
            config.DICE_WEIGHT * self.dice(y_pred, y_true)
            + config.FOCAL_WEIGHT * self.focal(y_pred, y_true)
            + config.CE_WEIGHT * self.ce(y_pred, y_true)
        )


def calculate_miou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> List[float]:
    """
    Calculates per-class Intersection over Union (IoU) scores.

    Args:
        y_true (np.ndarray): Flattened ground truth labels.
        y_pred (np.ndarray): Flattened predicted class labels.
        num_classes (int): Total number of target classes.

    Returns:
        List[float]: A list of IoU scores per class. Returns NaN if a class is not present.
    """
    ious: List[float] = []
    for c in range(num_classes):
        intersection = np.logical_and(y_true == c, y_pred == c).sum()
        union = np.logical_or(y_true == c, y_pred == c).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection / union))
    return ious


def calculate_iou_stats(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes cumulative intersection and union statistics for validation loops.

    Args:
        y_true (np.ndarray): Ground truth class labels.
        y_pred (np.ndarray): Predicted class labels.
        num_classes (int): Total number of architectural classes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (intersections, unions) arrays of shape (num_classes,).
    """
    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    for c in range(num_classes):
        intersections[c] = np.logical_and(y_true == c, y_pred == c).sum()
        unions[c] = np.logical_or(y_true == c, y_pred == c).sum()
    return intersections, unions
