import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from scipy.ndimage import maximum_filter

sys.path.append('.')
from models.heatmap_model_v4 import HeatmapDetectorV4
from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset

def analyze_sample(model, dataset, idx):
    device = torch.device("cuda")
    sample = dataset[idx]
    img_tensor = sample["frame"]
    gt_heatmap = sample["heatmap"]

    with torch.no_grad():
        pred_heatmap, _, _ = model(img_tensor.unsqueeze(0).to(device))

    pred = torch.sigmoid(pred_heatmap[0,0]).cpu().numpy()
    gt = gt_heatmap[0].numpy()

    print(f"\n{'='*60}")
    print(f"Sample {idx} — Heatmap Statistics")
    print(f"{'='*60}")
    print(f"Prediction max:        {pred.max():.4f}")
    print(f"Prediction mean:       {pred.mean():.4f}")
    print(f"Prediction std:        {pred.std():.4f}")
    print(f"Prediction > 0.1:      {(pred > 0.1).sum()}")
    print(f"Prediction > 0.3:      {(pred > 0.3).sum()}")
    print(f"Prediction > 0.5:      {(pred > 0.5).sum()}")

    print(f"\nGT max:                {gt.max():.4f}")
    print(f"GT mean:               {gt.mean():.4f}")
    print(f"GT > 0.5 pixels:       {(gt > 0.5).sum()}")

    # GT local maxima
    local_max_gt = maximum_filter(gt, size=5) == gt
    gt_peaks = np.argwhere((gt > 0.5) & local_max_gt)
    print(f"\nGT object centers: {len(gt_peaks)}")
    for cy, cx in gt_peaks[:10]:
        pred_val = pred[cy, cx]
        print(f"  GT at ({cx*4}, {cy*4}) -> pred value {pred_val:.4f}")

    # Pred local maxima at various thresholds
    local_max_pred = maximum_filter(pred, size=5) == pred
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        peaks = np.argwhere((pred > th) & local_max_pred)
        print(f"Pred local maxima > {th}: {len(peaks)}")

    # Save visualizations
    img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Draw GT centers red
    for cy, cx in gt_peaks:
        cv2.circle(img_color, (int(cx*4), int(cy*4)), 10, (0,0,255), 2)

    # Draw top 20 pred peaks at threshold 0.2
    pred_peaks = np.argwhere((pred > 0.2) & local_max_pred)
    # sort by pred value
    pred_peaks = sorted(pred_peaks, key=lambda p: pred[p[0], p[1]], reverse=True)[:20]
    for cy, cx in pred_peaks:
        cv2.circle(img_color, (int(cx*4), int(cy*4)), 8, (0,255,0), 2)

    out = f"diagnose_sample_{idx}.png"
    cv2.imwrite(out, img_color)
    print(f"\nSaved {out} (red=GT, green=top20 pred peaks)")

def main():
    device = torch.device("cuda")
    model = HeatmapDetectorV4(input_channels=1, output_stride=4).to(device)
    for path in ["heatmap_detector_1024_best.pth", "heatmap_detector_1024_improved_best.pth"]:
        if Path(path).exists():
            print(f"Loading: {path}")
            model.load_state_dict(torch.load(path, map_location=device))
            break
    model.eval()

    data_root = Path.home() / "space-debris-ai" / "data" / "v4"
    val_dataset = SpaceDebrisHeatmap1024Dataset(data_root, split="val", output_stride=4)

    for idx in range(3):
        analyze_sample(model, val_dataset, idx)

if __name__ == "__main__":
    main()
