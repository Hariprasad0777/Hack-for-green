"""
Modular Model Factory for Offroad Segmentation.
Supports UNet, UNet++, and DeepLabV3+ architectures via segmentation-models-pytorch.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Optional

def get_model(
    architecture: str = "unet", 
    encoder_name: str = "resnet34", 
    num_classes: int = 6,
    encoder_weights: str = "imagenet"
) -> nn.Module:
    """
    Factory function to initialize a segmentation model.

    Args:
        architecture (str): Type of architecture ('unet', 'deeplabv3+', 'unetplusplus').
        encoder_name (str): Name of the backbone encoder (e.g., 'resnet34', 'resnet50').
        num_classes (int): Number of output segmentation classes.
        encoder_weights (str): Pretrained weights to use (default: 'imagenet').

    Returns:
        nn.Module: Initialized PyTorch model.
    """
    arch_lower = architecture.lower()
    
    if arch_lower == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    elif arch_lower == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    elif arch_lower == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose from [unet, deeplabv3+, unetplusplus]")
    
    return model

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Smoke test model creation
    try:
        test_model = get_model()
        logger.info("[SUCCESS] Model factory initialized correctly.")
    except Exception as e:
        logger.error(f"[FAILURE] Model factory error: {e}")
