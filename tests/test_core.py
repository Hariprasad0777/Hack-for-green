import sys
import os
import torch
import numpy as np
import pytest

# Add project root to path


from offroad_ai.core.models import get_model
from offroad_ai.core.utils import calculate_miou, UltimateHybridLoss
from offroad_ai.core import config

def test_model_output_shape():
    """Verify that the model produces the correct output spatial dimensions."""
    model = get_model(architecture="unet", encoder_name="resnet18", num_classes=config.NUM_CLASSES)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    assert output.shape == (1, config.NUM_CLASSES, 256, 256)

def test_miou_calculation_perfect():
    """Verify mIoU calculation for perfect prediction."""
    y_true = np.array([[0, 1], [2, 0]])
    y_pred = np.array([[0, 1], [2, 0]])
    ious = calculate_miou(y_true, y_pred, num_classes=3)
    assert ious == [1.0, 1.0, 1.0]

def test_miou_calculation_zero():
    """Verify mIoU calculation for zero intersection."""
    y_true = np.array([[0, 0], [0, 0]])
    y_pred = np.array([[1, 1], [1, 1]])
    ious = calculate_miou(y_true, y_pred, num_classes=2)
    assert ious == [0.0, 0.0]

def test_hybrid_loss_execution():
    """Verify that the hybrid loss forward pass completes without error."""
    criterion = UltimateHybridLoss()
    y_pred = torch.randn(1, config.NUM_CLASSES, 64, 64)
    y_true = torch.randint(0, config.NUM_CLASSES, (1, 64, 64))
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

from offroad_ai.core.ensemble import TerrainEnsemble

def test_ensemble_initialization():
    """Verify that the ensemble initializes multiple models correctly."""
    model_configs = [
        {'arch': 'unet', 'encoder': 'resnet18', 'weight': 1.0},
        {'arch': 'unet', 'encoder': 'resnet34', 'weight': 0.5}
    ]
    ensemble = TerrainEnsemble(model_configs)
    assert len(ensemble.models) == 2
    assert ensemble.weights == [1.0, 0.5]

def test_ensemble_forward():
    """Verify that the ensemble produces a valid probability distribution."""
    model_configs = [
        {'arch': 'unet', 'encoder': 'resnet18', 'weight': 1.0},
    ]
    ensemble = TerrainEnsemble(model_configs)
    dummy_input = torch.randn(1, 3, 128, 128)
    output_probs = ensemble(dummy_input)
    # Output should be probabilities (sum to 1 over class dimension)
    probs_sum = torch.sum(output_probs, dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum))
