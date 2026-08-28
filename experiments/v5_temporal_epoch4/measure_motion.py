import numpy as np
from pathlib import Path
from models.dataset import SpaceDebrisDataset

ROOT = Path.home() / "space-debris-ai" / "data" / "v4"

dataset = SpaceDebrisDataset(
    ROOT,
    split="val",
    sequence_length=16
)

print("=" * 80)
print("TEMPORAL MOTION DIAGNOSTIC")
print("=" * 80)

all_displacements = []
all_dx = []
all_dy = []

for seq_idx in range(len(dataset)):

    sample = dataset[seq_idx]
    targets = sample["targets"]

    # Centers in image pixels for every frame
    frame_centers = []

    for frame_labels in targets:
        centers = []

        for obj in frame_labels:
            if len(obj) >= 5:
                cx = float(obj[1]) * 1024.0
                cy = float(obj[2]) * 1024.0
                centers.append((cx, cy))

        frame_centers.append(centers)

    # Match each object to its nearest object in the next frame.
    for t in range(len(frame_centers) - 1):

        a = frame_centers[t]
        b = frame_centers[t + 1]

        if not a or not b:
            continue

        used = set()

        for ax, ay in a:

            best = None
            best_dist = float("inf")

            for j, (bx, by) in enumerate(b):

                if j in used:
                    continue

                dx = bx - ax
                dy = by - ay

                dist = np.hypot(dx, dy)

                if dist < best_dist:
                    best_dist = dist
                    best = (j, dx, dy)

            if best is not None:
                j, dx, dy = best

                # Ignore obviously bad cross-matches
                if best_dist < 100:
                    used.add(j)

                    all_displacements.append(best_dist)
                    all_dx.append(dx)
                    all_dy.append(dy)

    if (seq_idx + 1) % 10 == 0:
        print(
            f"Processed {seq_idx + 1}/{len(dataset)} sequences"
        )

d = np.asarray(all_displacements)
dx = np.asarray(all_dx)
dy = np.asarray(all_dy)

print()
print("=" * 80)
print("FRAME-TO-FRAME DISPLACEMENT")
print("=" * 80)

print(f"Matches: {len(d)}")

if len(d):

    print()
    print("Displacement in image pixels:")
    print(f"Mean   : {d.mean():.4f}")
    print(f"Median : {np.median(d):.4f}")
    print(f"Std    : {d.std():.4f}")
    print(f"Min    : {d.min():.4f}")
    print(f"Max    : {d.max():.4f}")

    print()
    print("Percentiles:")

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(
            f"{p:3d}% : {np.percentile(d, p):.4f} px"
        )

    print()
    print("Fraction exceeding motion thresholds:")

    for threshold in [0.25, 0.5, 1, 2, 3, 5, 10]:
        print(
            f"> {threshold:4.2f} px : "
            f"{np.mean(d > threshold) * 100:7.2f}%"
        )

    print()
    print("dx / dy:")
    print(f"Mean |dx| : {np.mean(np.abs(dx)):.4f}")
    print(f"Mean |dy| : {np.mean(np.abs(dy)):.4f}")

else:
    print("No valid frame-to-frame matches found.")
