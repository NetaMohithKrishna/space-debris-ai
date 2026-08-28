import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter
from models.heatmap_model_v5_hr import HeatmapDetectorV5HR
from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset

DEVICE = torch.device("cuda")
STRIDE = 4

# Detection settings
THRESHOLD = 0.50
NMS_RADIUS = 5          # output pixels
MATCH_RADIUS = 5        # output pixels


def get_peaks(heatmap, threshold):
    local_max = maximum_filter(heatmap, size=5) == heatmap
    coords = np.argwhere(
        (heatmap >= threshold) & local_max
    )

    # Sort by confidence
    coords = sorted(
        coords,
        key=lambda p: heatmap[p[0], p[1]],
        reverse=True
    )

    # Simple NMS
    selected = []

    for y, x in coords:
        keep = True

        for sy, sx in selected:
            if (x - sx)**2 + (y - sy)**2 <= NMS_RADIUS**2:
                keep = False
                break

        if keep:
            selected.append((y, x))

    return selected


def get_gt_centers(gt):
    local_max = maximum_filter(gt, size=5) == gt

    coords = np.argwhere(
        (gt >= 0.5) & local_max
    )

    return [(y, x) for y, x in coords]


def match_predictions(preds, gts):
    matched_gt = set()
    tp = 0

    for py, px in preds:

        best = None
        best_dist = float("inf")

        for i, (gy, gx) in enumerate(gts):

            if i in matched_gt:
                continue

            dist = np.sqrt(
                (px - gx)**2 +
                (py - gy)**2
            )

            if dist <= MATCH_RADIUS and dist < best_dist:
                best_dist = dist
                best = i

        if best is not None:
            matched_gt.add(best)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp

    return tp, fp, fn


def main():

    model = HeatmapDetectorV5HR(
        input_channels=1,
        output_stride=4
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "heatmap_detector_1024_v5_hr_best.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    data_root = Path.home() / "space-debris-ai" / "data" / "v4"

    dataset = SpaceDebrisHeatmap1024Dataset(
        data_root,
        split="val",
        output_stride=4
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 70)
    print("OBJECT-LEVEL EVALUATION")
    print("=" * 70)

    for idx in range(len(dataset)):

        sample = dataset[idx]

        frame = sample["frame"]
        gt = sample["heatmap"][0].numpy()

        with torch.no_grad():
            pred, _, _ = model(
                frame.unsqueeze(0).to(DEVICE)
            )

        pred = torch.sigmoid(
            pred[0, 0]
        ).cpu().numpy()

        predictions = get_peaks(
            pred,
            THRESHOLD
        )

        gts = get_gt_centers(gt)

        tp, fp, fn = match_predictions(
            predictions,
            gts
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"Sample {idx:02d}: "
            f"GT={len(gts):2d} "
            f"Pred={len(predictions):3d} "
            f"TP={tp:2d} "
            f"FP={fp:3d} "
            f"FN={fn:2d}"
        )

    precision = total_tp / max(
        total_tp + total_fp, 1
    )

    recall = total_tp / max(
        total_tp + total_fn, 1
    )

    f1 = 2 * precision * recall / max(
        precision + recall, 1e-9
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"TP        : {total_tp}")
    print(f"FP        : {total_fp}")
    print(f"FN        : {total_fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")


if __name__ == "__main__":
    main()
