import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import time
import argparse
from offroad_ai.core import config

def simulate_feed(source_dir, stream_dir, interval=2.0):
    if not os.path.exists(stream_dir):
        os.makedirs(stream_dir)
    
    files = sorted([f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"Starting simulation. Copying {len(files)} files to {stream_dir} every {interval}s...")
    
    try:
        for f in files:
            src = os.path.join(source_dir, f)
            dst = os.path.join(stream_dir, f)
            shutil.copy2(src, dst)
            print(f"Stream: Added {f}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config.VAL_IMG_DIR)
    parser.add_argument("--stream", default="input_stream")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    
    simulate_feed(args.source, args.stream, args.interval)
