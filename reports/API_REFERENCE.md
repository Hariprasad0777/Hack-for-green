# 📖 API Reference: OffroadTerrain AI

## 1. Core Module (`core/`)

### `core.models.get_model`
**Signature**: `get_model(architecture: str, encoder_name: str, num_classes: int, encoder_weights: str) -> nn.Module`
Factory function for initializing SOTA segmentation models.
- **Args**:
  - `architecture`: One of `unet`, `deeplabv3+`, `unetplusplus`.
  - `encoder_name`: Backbone name (e.g., `resnet34`).
  - `num_classes`: Output channel count (default: 6).

### `core.ensemble.TerrainEnsemble`
**Signature**: `TerrainEnsemble(model_configs: List[Dict])`
Weighted average ensemble class for multi-backbone predictions.

### `core.utils.UltimateHybridLoss`
**Signature**: `UltimateHybridLoss(weights: torch.Tensor)`
Composite objective function: $0.4 \cdot Dice + 0.4 \cdot Focal + 0.2 \cdot CE$.

## 2. Pipeline Module (`pipeline/`)

### `pipeline.pathway_engine.perform_inference`
**Signature**: `perform_inference(image: bytes, model: nn.Module) -> np.ndarray`
High-performance inference wrapper for Pathway transformation nodes.

---
*Generated: 2026-02-27*
