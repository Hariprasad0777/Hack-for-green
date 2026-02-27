"""
Global Configuration for Offroad Terrain AI.

This module centralizes all hyper-parameters, file paths, and model 
architectural settings to ensure consistency across training, evaluation, 
and streaming pipelines.
"""

import os
import torch

# --- HYPER-PARAMETERS ---
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 4))          # Balanced for GPU memory at 512x512
EPOCHS: int = int(os.getenv("EPOCHS", 60))                # Total training epochs for full convergence
LR: float = float(os.getenv("LEARNING_RATE", 7e-5))       # Learning rate for AdamW optimizer
IMG_SIZE: tuple = (512, 512)                              # Input spatial resolution
NUM_CLASSES: int = 6                                      # Total classes (Bush, Grass, Rock, Water, Ground, Sky)

# --- ARCHITECTURE SETTINGS ---
# Verified architecture for the optimized competition checkpoint.
ARCHITECTURE: str = "unet" 
ENCODER: str = "resnet34"
ENCODER_WEIGHTS: str = "imagenet"

# --- LOSS FUNCTION COMPOSITION ---
DICE_WEIGHT: float = 0.4
FOCAL_WEIGHT: float = 0.4
CE_WEIGHT: float = 0.2

# --- DIRECTORY STRUCTURE ---
DATA_ROOT: str = "data"
TRAIN_IMG_DIR: str = os.path.join(DATA_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR: str = os.path.join(DATA_ROOT, "train", "Segmentation")
VAL_IMG_DIR: str = os.path.join(DATA_ROOT, "val", "Color_Images")
VAL_MASK_DIR: str = os.path.join(DATA_ROOT, "val", "Segmentation")

# --- ARTIFACTS & LOGGING ---
MODEL_SAVE_PATH: str = os.path.join("weights", "best_model.pth")
REPORTS_DIR: str = "reports"
PROMPT_LOG: str = os.path.join(REPORTS_DIR, "final_eval.txt")
VAL_LOG: str = os.path.join(REPORTS_DIR, "final_val.txt")

# --- COMPUTE INFRASTRUCTURE ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SEMANTIC MAPPING ---
# Logic: 0=Bush, 1=Grass, 2=Rock, 3=Water, 27=Ground, 39=Sky
CLASS_MAP: dict = {0: 0, 1: 1, 2: 2, 3: 3, 27: 4, 39: 5}
CLASS_NAMES: list = ["Bush", "Grass", "Rock", "Water", "Ground", "Sky"]

# --- CLASS BALANCING WEIGHTS ---
# Inverse pixel frequency derived weights for loss calculation.
# Sequence: [Bush, Grass, Rock, Water, Ground, Sky]
CLASS_WEIGHTS: torch.Tensor = torch.tensor([4.0, 2.0, 5.0, 15.0, 1.5, 1.0]).to(DEVICE)
