import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core.models import get_model
from offroad_ai.core import config

def apply_gaussian_smoothing(probs, sigma=0.5):
    """Applies Gaussian smoothing to the probability maps."""
    c, h, w = probs.shape
    smoothed = np.zeros_like(probs)
    for i in range(c):
        smoothed[i] = gaussian_filter(probs[i], sigma=sigma)
    return smoothed

def ultra_inference(model, image, scales=[0.7, 0.85, 1.0, 1.15, 1.3], biases=None, smoothing=0.5):
    """Dense TTA inference with Logit Biases and optional Smoothing."""
    _, _, h, w = image.shape
    device = image.device
    
    accum_probs = torch.zeros((1, config.NUM_CLASSES, h, w), device=device)
    
    with torch.no_grad():
        for scale in scales:
            for flip in [False, True]:
                img_scaled = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1.0 else image
                if flip:
                    img_scaled = torch.flip(img_scaled, dims=[3])
                
                logits = model(img_scaled)
                
                if biases is not None:
                    logits = logits + torch.from_numpy(biases).to(device).view(1, -1, 1, 1)
                
                probs = F.softmax(logits, dim=1)
                
                if flip:
                    probs = torch.flip(probs, dims=[3])
                
                if scale != 1.0:
                    probs = F.interpolate(probs, size=(h, w), mode='bilinear', align_corners=True)
                
                accum_probs += probs
                
    final_probs = (accum_probs / (len(scales) * 2)).cpu().numpy().squeeze(0)
    
    if smoothing > 0:
        final_probs = apply_gaussian_smoothing(final_probs, sigma=smoothing)
    
    return np.argmax(final_probs, axis=0)

def run_ultra_eval(bias_file="optimal_biases_full.npy"):
    print(f"[ULTRA] Starting Evaluation on {config.DEVICE}...")
    
    biases = np.load(bias_file) if os.path.exists(bias_file) else None
    if biases is not None:
        print(f"[+] Using Calibrated Biases: {biases.tolist()}")

    model = get_model(config.ARCHITECTURE, config.ENCODER, config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    val_ds = OffroadDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, 
                            transform=get_optimized_transforms(config.IMG_SIZE, is_train=False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    total_inter = np.zeros(config.NUM_CLASSES)
    total_union = np.zeros(config.NUM_CLASSES)
    
    for i, (images, masks) in enumerate(tqdm(val_loader, desc="Ultra Evaluating")):
        images = images.to(config.DEVICE)
        gt = masks.numpy().squeeze(0)
        
        # Dense inference
        preds = ultra_inference(model, images, biases=biases)
        
        for c in range(config.NUM_CLASSES):
            total_inter[c] += np.logical_and(gt == c, preds == c).sum()
            total_union[c] += np.logical_or(gt == c, preds == c).sum()
            
    ious = np.divide(total_inter, total_union, out=np.zeros_like(total_inter), where=total_union!=0)
    miou = np.nanmean(ious)
    
    print("\n--- ULTRA Evaluation Results ---")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:10}: {ious[i]:.4f}")
    print(f"OVERALL mIoU: {miou:.4f}")
    
    with open(config.VAL_LOG, "a") as f:
        f.write(f"\n[ULTRA EVAL] {datetime.now()} | mIoU: {miou:.4f}\n")
        for i, name in enumerate(config.CLASS_NAMES):
            f.write(f"  {name}: {ious[i]:.4f}\n")

if __name__ == "__main__":
    run_ultra_eval()
