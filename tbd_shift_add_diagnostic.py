import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


ROOT = Path.home() / "space-debris-ai" / "data" / "v4"
SPLIT = "val"

SEQUENCE_LENGTH = 16
FRAME_INTERVAL = 0.00025

# We will test motion hypotheses in image pixels/frame.
VELOCITIES = [
    (0.0, 0.0),
    (0.25, 0.0),
    (0.5, 0.0),
    (1.0, 0.0),
    (1.5, 0.0),
    (0.0, 0.25),
    (0.0, 0.5),
    (0.0, 1.0),
    (0.0, 1.5),
    (0.5, 0.5),
    (1.0, 1.0),
    (1.5, 1.5),
    (-0.5, 0.5),
    (-1.0, 1.0),
    (0.5, -0.5),
    (1.0, -1.0),
]

RADIUS = 5


def load_gray(path):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(
            f"Cannot read {path}"
        )

    return img.astype(np.float32) / 255.0


def extract_seq(stem):
    m = re.match(
        r"(.+)_frame_(\d+)$",
        stem
    )

    if not m:
        raise RuntimeError(
            f"Cannot parse {stem}"
        )

    return m.group(1), int(m.group(2))


def shift_image(img, dx, dy):
    """
    Shift image by subpixel amount while keeping output size.
    """
    h, w = img.shape

    M = np.array(
        [
            [1.0, 0.0, -dx],
            [0.0, 1.0, -dy],
        ],
        dtype=np.float32
    )

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def patch_max(img, x, y, radius):
    h, w = img.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    return float(
        img[y0:y1, x0:x1].max()
    )


def patch_mean(img, x, y, radius):
    h, w = img.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    return float(
        img[y0:y1, x0:x1].mean()
    )


def main():

    print("=" * 80)
    print("TRACK-BEFORE-DETECT / SHIFT-AND-ADD DIAGNOSTIC")
    print("=" * 80)

    meta_dir = ROOT / SPLIT / "metadata"

    sequences = sorted(
        p.stem
        for p in meta_dir.glob("*.json")
    )

    print(
        f"Sequences: {len(sequences)}"
    )

    all_results = []

    for seq_idx, seq in enumerate(sequences):

        meta_path = meta_dir / f"{seq}.json"

        with open(meta_path) as f:
            meta = json.load(f)

        image_dir = ROOT / SPLIT / "images"

        frame_paths = [
            image_dir
            / f"{seq}_frame_{i:03d}.png"
            for i in range(SEQUENCE_LENGTH)
        ]

        frames = [
            load_gray(p)
            for p in frame_paths
        ]

        # Metadata gives physical object trajectories.
        objects = meta["objects"]

        frame_objects = meta["frames"]

        for vx, vy in VELOCITIES:

            # Align all frames to frame 0 under the
            # hypothesized constant velocity.
            acc = np.zeros_like(frames[0])

            for t, frame in enumerate(frames):

                shifted = shift_image(
                    frame,
                    vx * t,
                    vy * t
                )

                acc += shifted

            acc /= len(frames)

            # Evaluate around frame-0 GT positions.
            for obj_idx, obj in enumerate(
                frame_objects[0]["objects"]
            ):

                x = obj["x_pixels"]
                y = obj["y_pixels"]

                response = patch_max(
                    acc,
                    x,
                    y,
                    RADIUS
                )

                local_mean = patch_mean(
                    acc,
                    x,
                    y,
                    RADIUS
                )

                all_results.append({
                    "seq": seq,
                    "obj": obj_idx,
                    "vx": vx,
                    "vy": vy,
                    "response": response,
                    "mean": local_mean,
                    "physical_size_cm":
                        meta["objects"][obj_idx][
                            "physical_size_cm"
                        ],
                    "distance_km":
                        meta["objects"][obj_idx][
                            "distance_km"
                        ],
                })

        if (seq_idx + 1) % 10 == 0:
            print(
                f"Processed {seq_idx + 1}/"
                f"{len(sequences)} sequences"
            )

    print()
    print("=" * 80)
    print("BEST VELOCITY HYPOTHESES")
    print("=" * 80)

    for vx, vy in VELOCITIES:

        vals = [
            r["response"]
            for r in all_results
            if r["vx"] == vx and r["vy"] == vy
        ]

        vals = np.asarray(vals, dtype=np.float32)

        print(
            f"vx={vx:+.2f}, vy={vy:+.2f} "
            f"mean={vals.mean():.6f} "
            f"median={np.median(vals):.6f} "
            f"p95={np.percentile(vals,95):.6f} "
            f"max={vals.max():.6f}"
        )

    print()
    print("=" * 80)
    print("SIZE-DEPENDENT TBD RESPONSE")
    print("=" * 80)

    # Find best velocity globally.
    best_pair = None
    best_mean = -1.0

    for vx, vy in VELOCITIES:

        vals = [
            r["response"]
            for r in all_results
            if r["vx"] == vx and r["vy"] == vy
        ]

        mean_val = float(
            np.mean(vals)
        )

        if mean_val > best_mean:
            best_mean = mean_val
            best_pair = (vx, vy)

    print(
        f"Best global hypothesis: "
        f"vx={best_pair[0]:+.2f}, "
        f"vy={best_pair[1]:+.2f}"
    )

    vx_best, vy_best = best_pair

    for size in [1, 2, 5, 10]:

        vals = [
            r["response"]
            for r in all_results
            if (
                r["vx"] == vx_best
                and r["vy"] == vy_best
                and abs(r["physical_size_cm"] - size) < 1e-6
            )
        ]

        vals = np.asarray(
            vals,
            dtype=np.float32
        )

        print(
            f"{size:2d} cm: "
            f"N={len(vals):4d} "
            f"mean={vals.mean():.6f} "
            f"median={np.median(vals):.6f} "
            f">0.02={100*np.mean(vals > 0.02):6.2f}% "
            f">0.03={100*np.mean(vals > 0.03):6.2f}%"
        )


if __name__ == "__main__":
    main()
