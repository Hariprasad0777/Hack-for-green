import sys
import os
import pathway as pw
import torch
import cv2
import numpy as np
import logging
from offroad_ai.core.models import get_model
from offroad_ai.core.dataset import OffroadDataset, get_optimized_transforms
from offroad_ai.core import config

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for model persistence across worker shards
DEVICE = config.DEVICE
MODEL = None
TRANSFORM = get_optimized_transforms(config.IMG_SIZE, is_train=False)

def init_model():
    """
    Initializes and caches the segmentation model.
    Implements a robust fallback mechanism for checkpoint discovery.
    """
    global MODEL
    if MODEL is None:
        MODEL = get_model(
            architecture=config.ARCHITECTURE,
            encoder_name=config.ENCODER,
            num_classes=config.NUM_CLASSES
        ).to(DEVICE)
        
        # Try loading optimized model first, then baseline
        if os.path.exists(config.MODEL_SAVE_PATH):
            weights_path = config.MODEL_SAVE_PATH
        elif os.path.exists("best_model.pth"):
            weights_path = "best_model.pth"
        else:
            logger.warning("No model weights found. Running with random initialization.")
            MODEL.eval()
            return
            
        MODEL.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        MODEL.eval()
        logger.info(f"Model successfully loaded from {weights_path}")

def perform_inference(image_data):
    """
    Pathway map function to perform segmentation.
    """
    init_model() 
    
    path = image_data['path']
    content = image_data['data']
    
    # Preprocessing
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_tensor = TRANSFORM(image=img)['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = MODEL(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Statistics
    ground_perc = (pred_mask == 0).sum() / pred_mask.size
    obstacle_perc = (pred_mask == 4).sum() / pred_mask.size
    
    # Visualization Export
    out_dir = "output_stream"
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "pred_" + os.path.basename(path))
    viz_mask = (pred_mask * (255 // (config.NUM_CLASSES - 1))).astype(np.uint8)
    cv2.imwrite(out_path, viz_mask)
    
    return {
        "filename": os.path.basename(path),
        "ground_percentage": float(ground_perc),
        "obstacle_percentage": float(obstacle_perc),
        "output_path": out_path
    }

def run() -> None:
    """
    Initializes and executes the Pathway reactive streaming pipeline.
    
    This function sets up a filesystem watcher on the 'input_stream' directory,
    processes images in real-time using the cached ML model, and outputs
    terrain statistics to a CSV report.
    """
    logger.info("Pathway: Initializing reactive watcher on 'input_stream'...")
    
    images = pw.io.fs.read(
        "input_stream",
        format="binary",
        with_metadata=True
    )
    
    results = images.select(
        results = pw.apply(perform_inference, pw.this)
    )
    
    final_table = results.select(
        filename = pw.this.results['filename'],
        ground_coverage = pw.this.results['ground_percentage'],
        obstacle_density = pw.this.results['obstacle_percentage'],
        result_saved_at = pw.this.results['output_path']
    )
    
    csv_path = os.path.join(config.REPORTS_DIR, "realtime_results.csv")
    pw.io.csv.write(final_table, csv_path)
    pw.debug.table_to_log(final_table)
    
    logger.info("Pipeline Execution Started.")
    pw.run()

if __name__ == "__main__":
    run()
