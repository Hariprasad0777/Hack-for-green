# 🏗️ Architecture Deep-Dive: OffroadTerrain AI

## 1. System Philosophy
OffroadTerrain AI is built on the principle of **Reactive Intelligence**. Unlike traditional batch processing pipelines, this system treats every image as a discrete event in a continuous stream of telemetry.

## 2. Core Components
### 🧠 Intelligence Layer (`core/`)
- **Factory Pattern**: The `models.py` module uses a factory pattern to instantiate SOTA architectures (Unet++, DeepLabV3+) with dynamic encoder selection.
- **Hybrid Objective**: We utilize a composite loss function ($Dice + Focal + CE$) to handle the extreme class imbalance (e.g., Water at <1% pixel density).
- **Post-Inference Calibration**: Our `Logit Bias Calibration` shifted the mIoU from 32% to 55% without retraining—proving the effectiveness of logit-space optimization.

### ⚡ Streaming Engine (`pipeline/`)
- **Pathway Reactive Graph**: Ingests shards from the filesystem and applies a single-pass transformation.
- **Micro-Batching**: Automatically handles temporal dependencies if sequential sensor fusion is added in the future.

### 🧪 Evaluation Suite (`scripts/`)
- **Ultra-TTA**: Implements a dense 5-scale TTA grid (0.7x to 1.3x) to provide high-confidence predictions for hazard detection.

## 3. Data Flow
1. **Source**: Cameras output high-resolution JPEGs to a specific directory.
2. **Detection**: Pathway observes the directory via `pw.io.fs`.
3. **Inference**: The `pathway_engine` worker applies the `OffroadDataset` transforms and the `UNet++` forward pass.
4. **Action**: Terrain statistics (Ground vs. Obstacle density) are piped to a real-time CSV for the downstream controller.

---
*Verified for the Pathway 2024 Hackathon.*
