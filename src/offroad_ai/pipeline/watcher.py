import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import cv2
import numpy as np
import pandas as pd
from offroad_ai.core.models import get_model
from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core import config

# Parameters
INPUT_DIR = "input_stream"
OUTPUT_DIR = "output_stream"
RESULTS_CSV = os.path.join(config.REPORTS_DIR, "realtime_results_windows.csv")
CHECK_INTERVAL = 1.0 # Seconds

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialization
DEVICE = config.DEVICE
TRANSFORM = get_optimized_transforms(config.IMG_SIZE, is_train=False)
model = None

def load_model():
    global model
    model = get_model(
        architecture=config.ARCHITECTURE,
        encoder_name=config.ENCODER,
        num_classes=config.NUM_CLASSES
    ).to(DEVICE)
    
    weights_path = config.MODEL_SAVE_PATH if os.path.exists(config.MODEL_SAVE_PATH) else "best_model.pth"
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print(f"Model loaded: {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path} (likely architecture mismatch or training in progress). Running with random initialization for now.")
    else:
        print("Warning: No checkpoint found. Running with random initialization.")
    model.eval()

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_tensor = TRANSFORM(image=img)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Stats
    ground_perc = (pred_mask == 0).sum() / pred_mask.size
    obstacle_perc = (pred_mask == 4).sum() / pred_mask.size
    
    # Save visualization
    out_filename = "pred_" + os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    viz_mask = (pred_mask * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
    cv2.imwrite(out_path, viz_mask)
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": os.path.basename(img_path),
        "ground_coverage": float(ground_perc),
        "obstacle_density": float(obstacle_perc),
        "result_path": out_path
    }

def start_watching():
    print(f"--- Windows Real-Time Engine Started ---")
    print(f"Watching: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    load_model()
    processed_files = set()
    results = []

    try:
        while True:
            current_files = {f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg'))}
            new_files = current_files - processed_files
            
            if new_files:
                for f in sorted(list(new_files)):
                    f_path = os.path.join(INPUT_DIR, f)
                    print(f"[*] New image detected: {f}")
                    
                    # Small delay to ensure file write is complete
                    time.sleep(0.5)
                    
                    data = process_image(f_path)
                    results.append(data)
                    processed_files.add(f)
                    
                    # Update local CSV
                    df = pd.DataFrame(results)
                    df.to_csv(RESULTS_CSV, index=False)
                    
                    print(f"    - Ground: {data['ground_coverage']:.2%}, Obstacles: {data['obstacle_density']:.2%}")
                    print(f"    - Result saved to {data['result_path']}")
            
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")

if __name__ == "__main__":
    start_watching()
