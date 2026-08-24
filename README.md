# Space Debris AI

A lightweight CNN-based space debris detection system designed for detecting small debris signatures in simulated satellite imagery.

## Project Structure

- `models/heatmap_model_v4.py` — CNN detector architecture
- `models/dataset_1024.py` — 1024×1024 heatmap target generation
- `scripts/physics_dataset_v4.py` — synthetic physics-inspired dataset generator
- `training/train_1024_improved.py` — training pipeline
- `evaluate_objects.py` — object-level evaluation
- `inspect_predictions.py` — prediction diagnostics
- `diagnose_heatmap.py` — heatmap diagnostics
- `diagnose_losses.py` — loss diagnostics
- `heatmap_detector_1024_improved_best.pth` — trained model checkpoint

## Model

Input:
1024×1024 grayscale image

Output:
256×256 heatmap

Output stride:
4

The network predicts:

1. Object-center heatmap
2. Center offsets
3. Object sizes

## Dataset

The project uses a physics-inspired synthetic dataset containing simulated debris, stars, optical PSF effects, motion during exposure, and sensor noise.

The dataset itself is not included in this repository.

## Training

```bash
PYTHONPATH=. python training/train_1024_improved.py
PY
py
