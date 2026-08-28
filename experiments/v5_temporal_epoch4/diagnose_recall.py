import torch
import numpy as np
from pathlib import Path
from models.heatmap_model_v5_temporal import HeatmapDetectorV5Temporal
from models.dataset_1024_temporal import SpaceDebrisTemporal1024Dataset
from scipy.ndimage import maximum_filter

DEVICE = torch.device("cuda")
MATCH_RADIUS = 5

model = HeatmapDetectorV5Temporal(output_stride=4).to(DEVICE)

model.load_state_dict(
    torch.load(
        "heatmap_detector_v5_temporal_best.pth",
        map_location=DEVICE
    )
)

model.eval()

data_root = Path.home() / "space-debris-ai" / "data" / "v4"

dataset = SpaceDebrisTemporal1024Dataset(
    data_root,
    split="val",
    output_stride=4
)

all_gt_scores = []
all_nearby_scores = []
all_max_scores = []

print("=" * 80)
print("GT RESPONSE DIAGNOSTIC")
print("=" * 80)

with torch.no_grad():

    for idx in range(len(dataset)):

        sample = dataset[idx]

        frame = sample["frame"]
        gt = sample["heatmap"][0].numpy()

        pred, _, _ = model(
            frame.unsqueeze(0).to(DEVICE)
        )

        pred = torch.sigmoid(
            pred[0, 0]
        ).cpu().numpy()

        # GT centers
        local_max = maximum_filter(gt, size=5) == gt

        gts = np.argwhere(
            (gt >= 0.5) & local_max
        )

        for gy, gx in gts:

            # Prediction exactly at GT
            exact = pred[gy, gx]

            # Maximum prediction in neighborhood
            y0 = max(0, gy - MATCH_RADIUS)
            y1 = min(pred.shape[0], gy + MATCH_RADIUS + 1)
            x0 = max(0, gx - MATCH_RADIUS)
            x1 = min(pred.shape[1], gx + MATCH_RADIUS + 1)

            nearby = pred[y0:y1, x0:x1]

            max_nearby = nearby.max()

            all_gt_scores.append(exact)
            all_nearby_scores.append(max_nearby)

        all_max_scores.append(pred.max())

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)}")

# Convert
gt_scores = np.array(all_gt_scores)
nearby_scores = np.array(all_nearby_scores)
max_scores = np.array(all_max_scores)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

print(f"Ground-truth objects : {len(gt_scores)}")

print()
print("Prediction at exact GT location:")
print(f"Mean   : {gt_scores.mean():.4f}")
print(f"Median : {np.median(gt_scores):.4f}")
print(f"Max    : {gt_scores.max():.4f}")
print(f"Min    : {gt_scores.min():.4f}")

print()
print("Maximum prediction within 5 pixels:")
print(f"Mean   : {nearby_scores.mean():.4f}")
print(f"Median : {np.median(nearby_scores):.4f}")
print(f"Max    : {nearby_scores.max():.4f}")
print(f"Min    : {nearby_scores.min():.4f}")

print()
print("GT objects recoverable at different scores:")

for t in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:

    exact_recall = np.mean(gt_scores >= t)
    nearby_recall = np.mean(nearby_scores >= t)

    print(
        f"{t:5.2f}   "
        f"exact={exact_recall*100:6.2f}%   "
        f"nearby={nearby_recall*100:6.2f}%"
    )

print()
print("=" * 80)
