"""
Ensemble module for Offroad Terrain AI.

This module provides the TerrainEnsemble class, which implements a weighted-average 
strategy to combine multiple segmentation model outputs into a single high-fidelity mask.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from offroad_ai.core import config
from offroad_ai.core.models import get_model


class TerrainEnsemble(nn.Module):
    """
    Weighted Average Ensemble for semantic segmentation backbones.

    This class allows for the combination of predictions from different architectures 
    (e.g., Unet++, DeepLabV3+) and encoders (e.g., ResNet34, ResNet50) to enhance 
    overall robustness and class-level accuracy.

    Attributes:
        models (nn.ModuleList): List of wrapped PyTorch models.
        weights (List[float]): Linear weights for each model's contribution.
    """

    def __init__(self, model_configs: List[Dict]):
        """
        Initializes the TerrainEnsemble with multiple model configurations.

        Args:
            model_configs (List[Dict]): A list of dictionaries, each containing:
                - 'arch' (str): Architecture name.
                - 'encoder' (str): Encoder/Backbone name.
                - 'path' (str, optional): Path to the model checkpoint.
                - 'weight' (float, optional): Voting weight for this model.
        """
        super().__init__()
        self.models = nn.ModuleList()
        self.weights: List[float] = []

        for cfg in model_configs:
            model = get_model(
                architecture=cfg["arch"],
                encoder_name=cfg["encoder"],
                num_classes=config.NUM_CLASSES,
            )
            if cfg.get("path"):
                model.load_state_dict(torch.load(cfg["path"], map_location="cpu"))
            self.models.append(model)
            self.weights.append(float(cfg.get("weight", 1.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes a multi-model forward pass and ensemble averaging.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Averaged probability distribution of shape (N, C, H, W).
        """
        total_probs: torch.Tensor = 0
        weight_sum = sum(self.weights)

        for i, model in enumerate(self.models):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            total_probs += probs * (self.weights[i] / weight_sum)

        return total_probs
