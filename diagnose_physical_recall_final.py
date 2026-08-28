import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from models.heatmap_model_v5_temporal import HeatmapDetectorV5Temporal
from models.dataset_1024_temporal import SpaceDebrisTemporal1024Dataset


DEVICE = torch.device("cuda")
ROOT = Path.home() / "space-debris-ai" / "data" / "v4"
CHECKPOINT = Path("heatmap_detector_v5_temporal_best.pth")

OUTPUT_STRIDE = 4
RADIUS = 5


def get_gt_centers(gt):
    local_max = maximum_filter(gt, size=5) == gt
    coords = np.argwhere((gt >= 0.5) & local_max)
    return [(int(y), int(x)) for y, x in coords]


def get_sequence_and_frame(stem):
    m = re.match(r"(.+)_frame_(\d+)$", stem)
    if not m:
        raise RuntimeError(f"Cannot parse frame name: {stem}")
    return m.group(1), int(m.group(2))


def load_metadata(seq):
    path = ROOT / "val" / "metadata" / f"{seq}.json"
    with open(path) as f:
        return json.load(f)


def nearest_prediction_score(pred, gy, gx, radius=5):
    h, w = pred.shape

    y0 = max(0, gy - radius)
    y1 = min(h, gy + radius + 1)
    x0 = max(0, gx - radius)
    x1 = min(w, gx + radius + 1)

    return float(pred[y0:y1, x0:x1].max())


def find_frame_object(frame_objects, gx, gy):
    """
    Match GT heatmap coordinate to frame-level metadata object.

    Metadata coordinates are image pixels.
    GT coordinates are output heatmap coordinates.
    """
    target_x = gx * OUTPUT_STRIDE
    target_y = gy * OUTPUT_STRIDE

    best_idx = None
    best_dist = float("inf")

    for i, obj in enumerate(frame_objects):
        x = float(obj["x_pixels"])
        y = float(obj["y_pixels"])

        dist = np.hypot(
            x - target_x,
            y - target_y
        )

        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx, best_dist


def physical_obj(seq_metadata, obj_idx):
    obj = seq_metadata["objects"][obj_idx]

    return {
        "size_cm": float(obj["physical_size_cm"]),
        "distance_km": float(obj["distance_km"]),
        "velocity": float(obj["velocity_mps"]),
        "reflectivity": float(obj["reflectivity"]),
        "illumination": float(obj["illumination"]),
    }


def main():

    print("=" * 80)
    print("V5-TEMPORAL PHYSICAL DETECTION DIAGNOSTIC")
    print("=" * 80)

    model = HeatmapDetectorV5Temporal(
        output_stride=4
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )
    )

    model.eval()

    dataset = SpaceDebrisTemporal1024Dataset(
        ROOT,
        split="val",
        output_stride=4
    )

    results = []
    bad_matches = 0

    for idx in range(len(dataset)):

        sample = dataset[idx]

        stem = dataset.images[idx].stem
        seq, frame_idx = get_sequence_and_frame(stem)

        metadata = load_metadata(seq)
        frame_objects = metadata["frames"][frame_idx]["objects"]

        gt = sample["heatmap"][0].numpy()

        with torch.no_grad():
            pred, _, _ = model(
                sample["frame"]
                .unsqueeze(0)
                .to(DEVICE)
            )

        pred = torch.sigmoid(
            pred[0, 0]
        ).cpu().numpy()

        gts = get_gt_centers(gt)

        for gy, gx in gts:

            obj_idx, image_dist = find_frame_object(
                frame_objects,
                gx,
                gy
            )

            # The image-space tolerance corresponding to the
            # 5-pixel output-space diagnostic radius.
            if obj_idx is None or image_dist > 5 * OUTPUT_STRIDE:
                bad_matches += 1
                continue

            physical = physical_obj(
                metadata,
                obj_idx
            )

            score = nearest_prediction_score(
                pred,
                gy,
                gx,
                RADIUS
            )

            frame_obj = frame_objects[obj_idx]

            results.append({
                **physical,
                "score": score,
                "geometric_pixels":
                    float(frame_obj["geometric_pixels"]),
                "apparent_width_pixels":
                    float(frame_obj["apparent_width_pixels"]),
                "apparent_height_pixels":
                    float(frame_obj["apparent_height_pixels"]),
                "motion_x_pixels":
                    float(frame_obj["motion_x_pixels"]),
                "motion_y_pixels":
                    float(frame_obj["motion_y_pixels"]),
                "motion_pixels":
                    float(
                        np.hypot(
                            frame_obj["motion_x_pixels"],
                            frame_obj["motion_y_pixels"]
                        )
                    ),
            })

        if (idx + 1) % 100 == 0:
            print(
                f"Processed {idx + 1}/{len(dataset)} frames"
            )

    print()
    print("=" * 80)
    print(f"Matched objects : {len(results)}")
    print(f"Bad GT matches  : {bad_matches}")
    print("=" * 80)

    def report(name, values):

        if not values:
            print(f"{name:<24} no samples")
            return

        scores = np.array(
            [r["score"] for r in values],
            dtype=np.float32
        )

        print(
            f"{name:<24}"
            f"N={len(values):4d} "
            f"mean={scores.mean():.3f} "
            f"median={np.median(scores):.3f} "
            f">0.30={100*np.mean(scores > 0.30):6.2f}% "
            f">0.50={100*np.mean(scores > 0.50):6.2f}% "
            f">0.60={100*np.mean(scores > 0.60):6.2f}%"
        )

    print("\nBY PHYSICAL SIZE")
    print("-" * 80)

    for size in [1, 2, 5, 10]:
        report(
            f"{size} cm",
            [
                r for r in results
                if abs(r["size_cm"] - size) < 1e-6
            ]
        )

    print("\nBY DISTANCE")
    print("-" * 80)

    for d in [0.5, 1, 2, 5, 10]:
        report(
            f"{d:g} km",
            [
                r for r in results
                if abs(r["distance_km"] - d) < 1e-6
            ]
        )

    print("\nBY APPARENT SIZE")
    print("-" * 80)

    bins = [
        ("<1 px", lambda r: r["apparent_width_pixels"] < 1),
        ("1-2 px", lambda r: 1 <= r["apparent_width_pixels"] < 2),
        ("2-4 px", lambda r: 2 <= r["apparent_width_pixels"] < 4),
        ("4-10 px", lambda r: 4 <= r["apparent_width_pixels"] < 10),
        (">=10 px", lambda r: r["apparent_width_pixels"] >= 10),
    ]

    for name, fn in bins:
        report(
            name,
            [r for r in results if fn(r)]
        )

    print("\nBY MOTION")
    print("-" * 80)

    motion_bins = [
        ("<0.25 px", lambda r: r["motion_pixels"] < 0.25),
        ("0.25-0.5", lambda r: 0.25 <= r["motion_pixels"] < 0.5),
        ("0.5-1 px", lambda r: 0.5 <= r["motion_pixels"] < 1),
        ("1-2 px", lambda r: 1 <= r["motion_pixels"] < 2),
        (">=2 px", lambda r: r["motion_pixels"] >= 2),
    ]

    for name, fn in motion_bins:
        report(
            name,
            [r for r in results if fn(r)]
        )

    print("\nBY REFLECTIVITY")
    print("-" * 80)

    if results:

        refl = np.array(
            [r["reflectivity"] for r in results]
        )

        q1, q2, q3 = np.percentile(
            refl,
            [25, 50, 75]
        )

        refl_bins = [
            ("Q1", lambda r: r["reflectivity"] <= q1),
            ("Q2", lambda r: q1 < r["reflectivity"] <= q2),
            ("Q3", lambda r: q2 < r["reflectivity"] <= q3),
            ("Q4", lambda r: r["reflectivity"] > q3),
        ]

        for name, fn in refl_bins:
            report(
                name,
                [r for r in results if fn(r)]
            )

    print("\nBY ILLUMINATION")
    print("-" * 80)

    illum_bins = [
        ("0-0.25", lambda r: r["illumination"] < 0.25),
        ("0.25-0.5", lambda r: 0.25 <= r["illumination"] < 0.5),
        ("0.5-0.75", lambda r: 0.5 <= r["illumination"] < 0.75),
        (">=0.75", lambda r: r["illumination"] >= 0.75),
    ]

    for name, fn in illum_bins:
        report(
            name,
            [r for r in results if fn(r)]
        )


if __name__ == "__main__":
    main()
