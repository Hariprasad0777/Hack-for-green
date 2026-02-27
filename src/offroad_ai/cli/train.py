"""
Training Module for Offroad Terrain AI.

This script implements the primary training loop, utilizing Mixed Precision (AMP),
Cosine Annealing, and the UltimateHybridLoss to achieve convergence on the 
offroad-segmentation task.
"""

import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from offroad_ai.core import config
from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core.models import get_model
from offroad_ai.core.utils import UltimateHybridLoss, calculate_miou

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(config.PROMPT_LOG), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train() -> None:
    """
    Executes the production training pipeline.

    This function initializes datasets, loaders, models, and optimizers, 
    then iterates through the training and validation epochs. It saves the 
    best model checkpoint based on mIoU.
    """
    logger.info("--- Production Training Pipeline Optimization ---")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Architecture: {config.ARCHITECTURE} | Encoder: {config.ENCODER}")

    # Ensure reports directory exists
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Datasets
    try:
        train_ds = OffroadDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=get_optimized_transforms(is_train=True),
        )
        val_ds = OffroadDataset(
            config.VAL_IMG_DIR,
            config.VAL_MASK_DIR,
            transform=get_optimized_transforms(is_train=False),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if config.DEVICE.type != "cpu" else False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )
    except Exception as e:
        logger.error(f"Failed to initialize datasets: {e}")
        return

    # Factory Model
    model = get_model(
        architecture=config.ARCHITECTURE,
        encoder_name=config.ENCODER,
        num_classes=config.NUM_CLASSES,
    ).to(config.DEVICE)

    criterion = UltimateHybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scaler: Optional[Any] = torch.amp.GradScaler("cuda") if config.DEVICE.type == "cuda" else None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_miou: float = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss: float = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss: float = 0.0
        all_ious: List[List[float]] = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]"):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gt = masks.cpu().numpy()

                for i in range(len(preds)):
                    all_ious.append(calculate_miou(gt[i], preds[i], config.NUM_CLASSES))

        mean_ious = np.nanmean(all_ious, axis=0)
        miou = np.nanmean(mean_ious)

        logger.info(f"Epoch {epoch+1} Summary: mIoU={miou:.4f}, Loss={val_loss/len(val_loader):.4f}")
        for i, name in enumerate(config.CLASS_NAMES):
            logger.info(f"  - {name}: {mean_ious[i]:.4f}")

        # Step Scheduler
        scheduler.step(miou)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(f"*** New Best mIoU ({best_miou:.4f})! Saved to {config.MODEL_SAVE_PATH} ***")


if __name__ == "__main__":
    train()
