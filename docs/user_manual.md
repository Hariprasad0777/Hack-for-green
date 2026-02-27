# 📖 User Manual: OffroadTerrain AI Operator's Guide

## 1. Introduction
OffroadTerrain AI provides real-time hazard detection for autonomous offroad vehicles. This guide outlines the operational steps for deployment.

## 2. Setting Up the Environment
Ensure Python 3.9+ and CUDA 11.8+ are installed.
```bash
make setup
```

## 3. Operations Workflow
### A. Initial Calibration
Before deployment, run the calibration suite to generate logit biases tailored to the specific local terrain (e.g., desert vs. forest).
```bash
python scripts/calibrate.py
```

### B. High-Fidelity Training
If new data is collected, fine-tune the model:
```bash
make train
```

### C. Live Mission Deployment
Launch the Pathway Engine to begin real-time segmentation:
```bash
make stream
```

## 4. Understanding Output
The system outputs a live stream of **Semantic Statistics** to `reports/realtime_results.csv`:
- `timestamp`: UTC time of inference.
- `ground_density`: Ratio of navigable area.
- `hazard_score`: Weighted urgency based on Rock/Water proximity.

---
*Autonomous Navigation Excellence.*
