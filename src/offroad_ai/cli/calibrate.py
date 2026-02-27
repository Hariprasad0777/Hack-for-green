import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core.models import get_model
from offroad_ai.core import config

def calibrate():
    device = config.DEVICE
    print(f"Starting Calibration on {device}...")
    
    model = get_model(architecture=config.ARCHITECTURE, encoder_name=config.ENCODER, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    val_ds = OffroadDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, 
                            transform=get_optimized_transforms(config.IMG_SIZE, is_train=False))
    # Sample a portion for faster calibration
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    num_samples = min(50, len(val_ds))
    
    all_logits = []
    all_gts = []
    
    print(f"Collecting logits for {num_samples} samples...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, total=num_samples)):
            if i >= num_samples: break
            images = images.to(device)
            logits = model(images) # [1, 6, H, W]
            all_logits.append(logits.cpu().numpy().squeeze(0))
            all_gts.append(masks.cpu().numpy().squeeze(0))

    def evaluate_biases(biases):
        total_inter = np.zeros(config.NUM_CLASSES)
        total_union = np.zeros(config.NUM_CLASSES)
        for logits, gt in zip(all_logits, all_gts):
            # Apply bias
            biased_logits = logits + biases[:, None, None]
            preds = np.argmax(biased_logits, axis=0)
            
            for c in range(config.NUM_CLASSES):
                total_inter[c] += np.logical_and(gt == c, preds == c).sum()
                total_union[c] += np.logical_or(gt == c, preds == c).sum()
        
        ious = np.divide(total_inter, total_union, out=np.zeros_like(total_inter), where=total_union!=0)
        return np.nanmean(ious), ious

    print("Search for optimal biases...")
    best_miou = 0
    best_biases = np.zeros(config.NUM_CLASSES)
    
    # Simple grid/random search for demonstration, then refine
    # Initial biases (Class 0, 2, 3 need a boost)
    current_biases = np.zeros(config.NUM_CLASSES)
    
    # Try boosting rare classes
    for c in [0, 2, 3]: # Bush, Rock, Water
        for step in np.linspace(0, 10, 21):
            temp_biases = current_biases.copy()
            temp_biases[c] = step
            miou, _ = evaluate_biases(temp_biases)
            if miou > best_miou:
                best_miou = miou
                best_biases = temp_biases
        current_biases = best_biases.copy()

    # Final refinement loop
    for _ in range(5):
        for c in range(config.NUM_CLASSES):
            for drift in [-0.5, -0.1, 0.1, 0.5]:
                temp_biases = best_biases.copy()
                temp_biases[c] += drift
                miou, _ = evaluate_biases(temp_biases)
                if miou > best_miou:
                    best_miou = miou
                    best_biases = temp_biases

    final_miou, final_ious = evaluate_biases(best_biases)
    print("\n--- Calibration Results ---")
    print(f"Best Biases: {best_biases.tolist()}")
    print(f"Calibrated mIoU (Sampled): {final_miou:.4f}")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:10}: {final_ious[i]:.4f}")
    
    # Save biases to a file for evaluate.py
    bias_path = os.path.join("weights", "optimal_biases.npy")
    np.save(bias_path, best_biases)
    print(f"\n[+] Biases saved to {bias_path}")

if __name__ == "__main__":
    calibrate()
