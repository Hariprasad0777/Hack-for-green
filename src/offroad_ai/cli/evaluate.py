"""
Evaluation Suite for Offroad Terrain AI.

This module provides high-density evaluation functions including Multi-Scale TTA,
Logit Bias Calibration application, and result visualization for the offroad 
segmentation task.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from offroad_ai.core import config
from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core.models import get_model
from offroad_ai.core.utils import calculate_iou_stats

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(config.VAL_LOG), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def inference_multi_scale_tta(
    model: nn.Module,
    image: torch.Tensor,
    scales: List[float] = [0.75, 1.0, 1.25],
    flip: bool = True,
    biases: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Performs multi-scale and horizontal-flip Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): The segmentation model.
        image (torch.Tensor): Input batch of shape (1, 3, H, W).
        scales (List[float]): Scales for interpolation.
        flip (bool): Whether to perform horizontal flip augmentation.
        biases (Optional[np.ndarray]): Class-specific logit biases.

    Returns:
        torch.Tensor: Averaged probability mask of shape (1, C, H, W).
    """
    _, _, h, w = image.shape
    device = image.device

    final_probs: torch.Tensor = torch.zeros((1, config.NUM_CLASSES, h, w), device=device)
    count: int = 0

    with torch.no_grad():
        for scale in scales:
            if scale == 1.0:
                img_scaled = image
            else:
                img_scaled = F.interpolate(
                    image, scale_factor=scale, mode="bilinear", align_corners=True
                )

            logits = model(img_scaled)

            # Apply biases if provided
            if biases is not None:
                logits = logits + torch.from_numpy(biases).to(device).view(1, -1, 1, 1)

            probs = F.softmax(logits, dim=1)

            if scale != 1.0:
                probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=True)

            final_probs += probs
            count += 1

            if flip:
                img_flipped = torch.flip(img_scaled, dims=[3])
                logits_flipped = model(img_flipped)

                if biases is not None:
                    logits_flipped = logits_flipped + torch.from_numpy(biases).to(device).view(
                        1, -1, 1, 1
                    )

                probs_flipped = F.softmax(logits_flipped, dim=1)
                probs_flipped = torch.flip(probs_flipped, dims=[3])

                if scale != 1.0:
                    probs_flipped = F.interpolate(
                        probs_flipped, size=(h, w), mode="bilinear", align_corners=True
                    )

                final_probs += probs_flipped
                count += 1

    return final_probs / count


def run_evaluation(use_tta: bool = True, bias_file: Optional[str] = None) -> None:
    """
    Executes a comprehensive evaluation on the validation set.

    Args:
        use_tta (bool): Whether to use Test-Time Augmentation.
        bias_file (Optional[str]): Path to calibrated logit bias .npy file.
    """
    logger.info(
        f"Starting Evaluation | Arch: {config.ARCHITECTURE} | TTA: {use_tta} | Bias: {bias_file}"
    )

    biases: Optional[np.ndarray] = None
    if bias_file and os.path.exists(bias_file):
        biases = np.load(bias_file)
        logger.info("Loaded Calibrated Biases.")

    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    weights_path: str = config.MODEL_SAVE_PATH
    if not os.path.exists(weights_path):
        logger.error(f"Checkpoint {weights_path} not found.")
        return

    logger.info(f"Loading Checkpoint: {weights_path}")
    model = get_model(
        architecture=config.ARCHITECTURE,
        encoder_name=config.ENCODER,
        num_classes=config.NUM_CLASSES,
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
    model.eval()

    val_ds = OffroadDataset(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        transform=get_optimized_transforms(is_train=False),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    total_intersections: np.ndarray = np.zeros(config.NUM_CLASSES)
    total_unions: np.ndarray = np.zeros(config.NUM_CLASSES)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(config.DEVICE)

            if use_tta:
                probs = inference_multi_scale_tta(model, images, biases=biases)
            else:
                outputs = model(images)
                if biases is not None:
                    outputs = outputs + torch.from_numpy(biases).to(config.DEVICE).view(1, -1, 1, 1)
                probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1).cpu().numpy().squeeze(0)
            gt = masks.cpu().numpy().squeeze(0)

            inter, union = calculate_iou_stats(gt, preds, config.NUM_CLASSES)
            total_intersections += inter
            total_unions += union

    ious: np.ndarray = np.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        if total_unions[c] > 0:
            ious[c] = total_intersections[c] / total_unions[c]
        else:
            ious[c] = np.nan

    mean_iou: float = float(np.nanmean(ious))

    logger.info("\n--- Final Evaluation Results ---")
    logger.info(f"Checkpoint: {weights_path}")
    logger.info(f"TTA Enabled: {use_tta}")
    for i, name in enumerate(config.CLASS_NAMES):
        logger.info(f"  {name:10}: {ious[i]:.4f}")
    logger.info(f"Overall mIoU: {mean_iou:.4f}")

    output_file: str = config.VAL_LOG
    suffix: str = " (Calibrated)" if biases is not None else ""
    with open(output_file, "a") as f:
        f.write(f"\n[ENHANCED EVALUATION{suffix}] {timestamp}\n")
        f.write(
            f"Checkpoint: {weights_path} | Arch: {config.ARCHITECTURE} | TTA: {use_tta} | mIoU: {mean_iou:.4f}\n"
        )
        for i, name in enumerate(config.CLASS_NAMES):
            f.write(f"  {name}: {ious[i]:.4f}\n")

    logger.info(f"Results archived in {output_file}")


def visualize_inference(
    num_samples: int = 3, use_tta: bool = True, bias_file: Optional[str] = None
) -> None:
    """
    Generates side-by-side visual comparisons of Image vs GT vs Pred.

    Args:
        num_samples (int): Number of images to visualize.
        use_tta (bool): Whether to use TTA in the visualization.
        bias_file (Optional[str]): Path to logic bias .npy file.
    """
    weights_path: str = config.MODEL_SAVE_PATH
    if not os.path.exists(weights_path):
        return

    biases: Optional[np.ndarray] = None
    if bias_file and os.path.exists(bias_file):
        biases = np.load(bias_file)

    model = get_model(
        architecture=config.ARCHITECTURE,
        encoder_name=config.ENCODER,
        num_classes=config.NUM_CLASSES,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
    model.eval()

    val_ds = OffroadDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, transform=None)
    transform = get_optimized_transforms(is_train=False)

    indices = np.random.choice(len(val_ds), min(num_samples, len(val_ds)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    colors: List[List[int]] = [
        [0, 255, 0],
        [0, 128, 0],
        [128, 128, 128],
        [255, 0, 0],
        [255, 255, 255],
        [0, 191, 255],
    ]

    for i, idx in enumerate(indices):
        img_path = os.path.join(config.VAL_IMG_DIR, val_ds.img_files[idx])
        display_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        input_tensor = transform(image=display_img)["image"].unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            if use_tta:
                probs = inference_multi_scale_tta(model, input_tensor, biases=biases)
            else:
                logits = model(input_tensor)
                if biases is not None:
                    logits = logits + torch.from_numpy(biases).to(config.DEVICE).view(1, -1, 1, 1)
                probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

        pred_mask_resized = cv2.resize(
            pred_mask, (display_img.shape[1], display_img.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        def colorize(mask: np.ndarray) -> np.ndarray:
            c_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for c in range(config.NUM_CLASSES):
                c_mask[mask == c] = colors[c]
            return c_mask

        _, raw_mask = val_ds[idx]
        gt_colored = colorize(raw_mask)
        pred_colored = colorize(pred_mask_resized)

        axes[i, 0].imshow(display_img)
        axes[i, 0].set_title(f"Image: {val_ds.img_files[idx]}")
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title(f"Pred (TTA={use_tta})")

    plt.tight_layout()
    viz_path: str = "reports/evaluation_visuals.png"
    plt.savefig(viz_path)
    logger.info(f"Visualizations saved to {viz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA for faster evaluation")
    parser.add_argument("--samples", type=int, default=3, help="Num visualization samples")
    parser.add_argument(
        "--bias-file", type=str, default="weights/optimal_biases.npy", help="Path to .npy bias file"
    )
    args = parser.parse_args()

    active_tta: bool = not args.no_tta
    run_evaluation(use_tta=active_tta, bias_file=args.bias_file)
    visualize_inference(num_samples=args.samples, use_tta=active_tta, bias_file=args.bias_file)
