import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter

from models.heatmap_model_v4 import HeatmapDetectorV4
from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset


DEVICE = torch.device("cuda")
MODEL_PATH = "heatmap_detector_1024_improved_best.pth"


model = HeatmapDetectorV4(
    input_channels=1,
    output_stride=4
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

root = Path.home() / "space-debris-ai" / "data" / "v4"

dataset = SpaceDebrisHeatmap1024Dataset(
    root,
    split="val",
    output_stride=4
)

sample = dataset[0]

frame = sample["frame"].unsqueeze(0).to(DEVICE)
gt = sample["heatmap"][0].numpy()

with torch.no_grad():
    pred, _, _ = model(frame)

raw = pred[0, 0].detach().cpu().numpy()
prob = torch.sigmoid(pred[0, 0]).detach().cpu().numpy()

print("=" * 70)
print("RAW LOGIT")
print("=" * 70)

print(f"min    : {raw.min():.6f}")
print(f"max    : {raw.max():.6f}")
print(f"mean   : {raw.mean():.6f}")
print(f"median : {np.median(raw):.6f}")
print(f"std    : {raw.std():.6f}")

print("\nPercentiles:")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"{p:3d}% : {np.percentile(raw, p):.6f}")


print("\n" + "=" * 70)
print("SIGMOID PROBABILITY")
print("=" * 70)

print(f"min    : {prob.min():.6f}")
print(f"max    : {prob.max():.6f}")
print(f"mean   : {prob.mean():.6f}")
print(f"median : {np.median(prob):.6f}")
print(f"std    : {prob.std():.6f}")

print("\nProbability counts:")
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.8, 0.9]:
    print(f"> {th:.3f}: {np.sum(prob > th)}")


print("\n" + "=" * 70)
print("LOCAL MAXIMA")
print("=" * 70)

local_max = maximum_filter(prob, size=5)

for th in [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.8, 0.9]:
    peaks = np.argwhere(
        (prob > th) &
        (local_max == prob)
    )

    print(f"peaks > {th:.3f}: {len(peaks)}")


print("\n" + "=" * 70)
print("GROUND TRUTH")
print("=" * 70)

print(f"GT min    : {gt.min():.6f}")
print(f"GT max    : {gt.max():.6f}")
print(f"GT mean   : {gt.mean():.6f}")
print(f"GT >0.5   : {np.sum(gt > 0.5)}")

gt_local = maximum_filter(gt, size=5)

gt_peaks = np.argwhere(
    (gt > 0.5) &
    (gt_local == gt)
)

print(f"GT peaks  : {len(gt_peaks)}")

for y, x in gt_peaks:
    print(
        f"GT peak: heatmap=({x},{y}) "
        f"image=({x*4},{y*4}) "
        f"value={gt[y,x]:.3f}"
    )
