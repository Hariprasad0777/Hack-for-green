"""
Offroad Terrain Dataset and Augmentation Pipeline.
Handles semantic mask remapping and robust imagery augmentations for offroad contexts.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, Any, Dict
from offroad_ai.core import config

class OffroadDataset(Dataset):
    """
    Custom Dataset for Offroad Terrain Segmentation.
    
    Attributes:
        img_dir (str): Path to color images.
        mask_dir (str): Path to segmentation masks.
        transform (A.Compose): Albumentations mapping for data augmentation.
        class_map (Dict): Mapping from raw pixel values to normalized class indices.
    """
    def __init__(self, img_dir: str, mask_dir: str, transform: Optional[A.Compose] = None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.class_map = config.CLASS_MAP
        
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single image-mask pair with robust remapping and transformations.
        """
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.img_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Robust class remapping (e.g., mapping obstacle codes to contiguous indices)
        remapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for old_val, new_val in self.class_map.items():
            remapped_mask[mask == old_val] = new_val
            
        if self.transform:
            augmented = self.transform(image=image, mask=remapped_mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            # Baseline preprocessing fallback
            image = cv2.resize(image, config.IMG_SIZE)
            mask = cv2.resize(remapped_mask, config.IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def get_optimized_transforms(img_size: Tuple[int, int] = config.IMG_SIZE, is_train: bool = True) -> A.Compose:
    """
    Standardized augmentation factory for offroad environments.
    
    Returns a pipeline optimized for varying illumination and terrain distortion.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            
            # Spatial Diversification
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            
            # Environmental Invariance
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.4),
            
            # Optical & Lens Distortions
            A.OneOf([
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=0.1, p=1),
            ], p=0.2),
            
            # Perceptual Noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.Sharpen(p=1),
            ], p=0.2),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
