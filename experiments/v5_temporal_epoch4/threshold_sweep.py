import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter
from models.heatmap_model_v5_temporal import HeatmapDetectorV5Temporal
from models.dataset_1024_temporal import SpaceDebrisTemporal1024Dataset

DEVICE = torch.device("cuda")
NMS_RADIUS = 5
MATCH_RADIUS = 5


def get_peaks(heatmap, threshold):
    local_max = maximum_filter(heatmap, size=5) == heatmap

    coords = np.argwhere(
        (heatmap >= threshold) & local_max
    )

    coords = sorted(
        coords,
        key=lambda p: heatmap[p[0], p[1]],
        reverse=True
    )

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
    local_max = maximum_filter(gt, size=5)

    coords = np.argwhere(
        (gt >= 0.5) & (local_max == gt)
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

    print("Loading model...")

    model = HeatmapDetectorV5Temporal(
        output_stride=4
    ).to(DEVICE)

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

    print(f"Validation samples: {len(dataset)}")
    print("Running inference once and storing predictions...")

    all_predictions = []
    all_gts = []

    # Run model ONLY ONCE per sample.
    # This makes the threshold sweep much faster.
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

        all_predictions.append(pred)
        all_gts.append(get_gt_centers(gt))

        print(f"Processed {idx + 1}/{len(dataset)}", end="\r")

    print("\n")

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    print("=" * 90)
    print(
        f"{'Threshold':>10} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )
    print("=" * 90)

    results = []

    for threshold in thresholds:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, gts in zip(all_predictions, all_gts):

            predictions = get_peaks(
                pred,
                threshold
            )

            tp, fp, fn = match_predictions(
                predictions,
                gts
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / max(
            total_tp + total_fp,
            1
        )

        recall = total_tp / max(
            total_tp + total_fn,
            1
        )

        f1 = 2 * precision * recall / max(
            precision + recall,
            1e-9
        )

        results.append(
            (
                threshold,
                total_tp,
                total_fp,
                total_fn,
                precision,
                recall,
                f1
            )
        )

        print(
            f"{threshold:10.2f} "
            f"{total_tp:8d} "
            f"{total_fp:8d} "
            f"{total_fn:8d} "
            f"{precision:12.4f} "
            f"{recall:10.4f} "
            f"{f1:10.4f}"
        )

    best = max(results, key=lambda x: x[-1])

    print("\n" + "=" * 90)
    print("BEST F1")
    print("=" * 90)

    print(
        f"Threshold : {best[0]:.2f}\n"
        f"TP        : {best[1]}\n"
        f"FP        : {best[2]}\n"
        f"FN        : {best[3]}\n"
        f"Precision : {best[4]:.4f}\n"
        f"Recall    : {best[5]:.4f}\n"
        f"F1        : {best[6]:.4f}"
    )


if __name__ == "__main__":
    main()
