import cv2
import numpy as np
from pathlib import Path
from models.dataset import SpaceDebrisDataset
from scipy.ndimage import maximum_filter

ROOT = Path.home() / "space-debris-ai" / "data" / "v4"
STRIDE = 4


def centers_from_labels(labels):
    pts = []

    for obj in labels:
        if len(obj) >= 5:
            pts.append((
                float(obj[1]) * 1024.0,
                float(obj[2]) * 1024.0
            ))

    return pts


def nearest_match(points_a, points_b, max_dist=40):
    matches = []
    used = set()

    for ax, ay in points_a:

        best = None
        best_dist = float("inf")

        for j, (bx, by) in enumerate(points_b):

            if j in used:
                continue

            d = np.hypot(
                bx - ax,
                by - ay
            )

            if d < best_dist:
                best_dist = d
                best = (j, bx, by)

        if best is not None and best_dist <= max_dist:
            j, bx, by = best
            used.add(j)
            matches.append(
                (ax, ay, bx, by, best_dist)
            )

    return matches


def patch_max(diff, x, y, radius=5):
    h, w = diff.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    return float(diff[y0:y1, x0:x1].max())


def patch_mean(diff, x, y, radius=4):
    h, w = diff.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    return float(diff[y0:y1, x0:x1].mean())


def background_values(diff, x, y):
    h, w = diff.shape

    x = int(round(x))
    y = int(round(y))

    # Sample an outer annulus around the target
    r1 = 8
    r2 = 15

    x0 = max(0, x - r2)
    x1 = min(w, x + r2 + 1)
    y0 = max(0, y - r2)
    y1 = min(h, y + r2 + 1)

    patch = diff[y0:y1, x0:x1]

    yy, xx = np.indices(patch.shape)

    cx = x - x0
    cy = y - y0

    d2 = (xx - cx) ** 2 + (yy - cy) ** 2

    bg = patch[
        (d2 >= r1 ** 2) &
        (d2 <= r2 ** 2)
    ]

    return bg


def main():

    dataset = SpaceDebrisDataset(
        ROOT,
        split="val",
        sequence_length=16
    )

    debris_max = []
    debris_mean = []
    debris_snr = []

    bg_max = []
    bg_mean = []

    for seq_idx in range(len(dataset)):

        sample = dataset[seq_idx]
        targets = sample["targets"]

        sequence = sample["sequence"]

        image_dir = ROOT / "val" / "images"

        files = sorted(
            image_dir.glob(
                f"{sequence}_frame_*.png"
            )
        )[:16]

        if len(files) < 2:
            continue

        frames = []

        for f in files:
            img = cv2.imread(
                str(f),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            frames.append(
                img.astype(np.float32) / 255.0
            )

        for t in range(len(frames) - 1):

            a = frames[t]
            b = frames[t + 1]

            diff = np.abs(b - a)

            pts_a = centers_from_labels(
                targets[t]
            )

            pts_b = centers_from_labels(
                targets[t + 1]
            )

            matches = nearest_match(
                pts_a,
                pts_b
            )

            for ax, ay, bx, by, displacement in matches:

                # The residual can appear around either
                # the old or new position.
                old_max = patch_max(
                    diff,
                    ax,
                    ay,
                    radius=6
                )

                new_max = patch_max(
                    diff,
                    bx,
                    by,
                    radius=6
                )

                response = max(
                    old_max,
                    new_max
                )

                old_mean = patch_mean(
                    diff,
                    ax,
                    ay
                )

                new_mean = patch_mean(
                    diff,
                    bx,
                    by
                )

                response_mean = max(
                    old_mean,
                    new_mean
                )

                bg = np.concatenate([
                    background_values(
                        diff,
                        ax,
                        ay
                    ),
                    background_values(
                        diff,
                        bx,
                        by
                    )
                ])

                bg_med = np.median(bg)
                bg_std = np.std(bg) + 1e-6

                snr = (
                    response - bg_med
                ) / bg_std

                debris_max.append(response)
                debris_mean.append(response_mean)
                debris_snr.append(snr)

                bg_max.append(
                    np.percentile(bg, 95)
                )
                bg_mean.append(
                    bg.mean()
                )

        if (seq_idx + 1) % 10 == 0:
            print(
                f"Processed "
                f"{seq_idx + 1}/{len(dataset)}"
            )

    debris_max = np.asarray(debris_max)
    debris_mean = np.asarray(debris_mean)
    debris_snr = np.asarray(debris_snr)

    bg_max = np.asarray(bg_max)
    bg_mean = np.asarray(bg_mean)

    print()
    print("=" * 80)
    print("TEMPORAL GT RESIDUAL DIAGNOSTIC")
    print("=" * 80)

    print(f"Objects matched: {len(debris_max)}")

    print()
    print("DEBRIS TEMPORAL RESIDUAL")
    print(f"Max response mean   : {debris_max.mean():.6f}")
    print(f"Max response median : {np.median(debris_max):.6f}")
    print(f"Mean patch response : {debris_mean.mean():.6f}")

    print()
    print("BACKGROUND")
    print(f"95% max background mean   : {bg_max.mean():.6f}")
    print(f"Background mean           : {bg_mean.mean():.6f}")

    print()
    print("TEMPORAL SNR")
    print(f"Mean   : {debris_snr.mean():.4f}")
    print(f"Median : {np.median(debris_snr):.4f}")

    print()
    for th in [1, 2, 3, 5, 10]:

        print(
            f"SNR > {th:2d}: "
            f"{np.mean(debris_snr > th) * 100:.2f}%"
        )

    print()
    print("Residual response thresholds")

    for th in [
        0.005,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050
    ]:

        print(
            f"> {th:.3f}: "
            f"{np.mean(debris_max > th) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
