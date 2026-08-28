import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "space-debris-ai" / "data" / "v4"
SPLIT = "val"

T = 16
STRIDE = 4


def load_gray(path):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    return img.astype(np.float32) / 255.0


def shift_image(img, dx, dy):
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


def patch_values(img, x, y, radius=5):
    h, w = img.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)

    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    return img[y0:y1, x0:x1]


def main():

    print("=" * 80)
    print("ORACLE TRACK-BEFORE-DETECT DIAGNOSTIC")
    print("=" * 80)

    meta_dir = ROOT / SPLIT / "metadata"
    image_dir = ROOT / SPLIT / "images"

    sequences = sorted(
        p.stem
        for p in meta_dir.glob("*.json")
    )

    results = []

    for seq_idx, seq in enumerate(sequences):

        with open(meta_dir / f"{seq}.json") as f:
            meta = json.load(f)

        frames = [
            load_gray(
                image_dir / f"{seq}_frame_{t:03d}.png"
            )
            for t in range(T)
        ]

        objects0 = meta["frames"][0]["objects"]

        for obj_idx, obj0 in enumerate(objects0):

            x0 = float(obj0["x_pixels"])
            y0 = float(obj0["y_pixels"])

            # Use the actual trajectory from metadata.
            # Objects may leave the field of view, so only use
            # frames where this object is still present.
            valid_t = []

            for t in range(T):
                frame_objects = meta["frames"][t]["objects"]

                if obj_idx < len(frame_objects):
                    valid_t.append(t)

            if len(valid_t) < 2:
                continue

            xs = np.array([
                meta["frames"][t]["objects"][obj_idx]["x_pixels"]
                for t in valid_t
            ])

            ys = np.array([
                meta["frames"][t]["objects"][obj_idx]["y_pixels"]
                for t in valid_t
            ])

            # Rebase trajectory so the first valid frame is the origin.
            x_ref = xs[0]
            y_ref = ys[0]

            dx = xs - x_ref
            dy = ys - y_ref

            # ---------------------------------------------------------
            # 1. Raw frame-0 local signal
            # ---------------------------------------------------------
            raw_patch = patch_values(
                frames[valid_t[0]],
                x_ref,
                y_ref
            )

            raw_signal = float(raw_patch.max())

            # ---------------------------------------------------------
            # 2. Temporal difference along true trajectory
            # ---------------------------------------------------------
            diff_acc = np.zeros_like(frames[0])

            for k in range(1, len(valid_t)):

                t_prev = valid_t[k - 1]
                t_curr = valid_t[k]

                diff = np.abs(
                    frames[t_curr] - frames[t_prev]
                )

                aligned = shift_image(
                    diff,
                    dx[k],
                    dy[k]
                )

                diff_acc += aligned

            diff_acc /= max(len(valid_t) - 1, 1)

            diff_patch = patch_values(
                diff_acc,
                x_ref,
                y_ref
            )

            diff_signal = float(
                diff_patch.max()
            )

            # ---------------------------------------------------------
            # 3. Raw-frame oracle shift-and-add
            # ---------------------------------------------------------
            raw_acc = np.zeros_like(frames[0])

            for k, t in enumerate(valid_t):

                aligned = shift_image(
                    frames[t],
                    dx[k],
                    dy[k]
                )

                raw_acc += aligned

            raw_acc /= len(valid_t)

            oracle_patch = patch_values(
                raw_acc,
                x_ref,
                y_ref
            )

            oracle_signal = float(
                oracle_patch.max()
            )

            size_cm = float(
                meta["objects"][obj_idx]["physical_size_cm"]
            )

            distance_km = float(
                meta["objects"][obj_idx]["distance_km"]
            )

            velocity = float(
                meta["objects"][obj_idx]["velocity_mps"]
            )

            motion_per_frame = float(
                np.hypot(dx[1], dy[1])
            )

            total_motion = float(
                np.hypot(dx[-1], dy[-1])
            )

            results.append({
                "size": size_cm,
                "distance": distance_km,
                "velocity": velocity,
                "motion": motion_per_frame,
                "total_motion": total_motion,
                "raw": raw_signal,
                "diff": diff_signal,
                "oracle": oracle_signal,
            })

        if (seq_idx + 1) % 10 == 0:
            print(
                f"Processed {seq_idx + 1}/{len(sequences)} sequences"
            )

    print()
    print("=" * 80)
    print("GLOBAL RESULTS")
    print("=" * 80)

    for key in ["raw", "diff", "oracle"]:

        values = np.array(
            [r[key] for r in results],
            dtype=np.float32
        )

        print(
            f"{key:8s}: "
            f"mean={values.mean():.6f} "
            f"median={np.median(values):.6f} "
            f"p95={np.percentile(values,95):.6f}"
        )

    print()
    print("=" * 80)
    print("BY SIZE")
    print("=" * 80)

    for size in [1, 2, 5, 10]:

        vals = [
            r for r in results
            if abs(r["size"] - size) < 1e-6
        ]

        print(f"\n{size} cm  N={len(vals)}")

        for key in ["raw", "diff", "oracle"]:

            values = np.array(
                [r[key] for r in vals],
                dtype=np.float32
            )

            print(
                f"  {key:8s} "
                f"mean={values.mean():.6f} "
                f"median={np.median(values):.6f}"
            )

    print()
    print("=" * 80)
    print("BY MOTION")
    print("=" * 80)

    bins = [
        ("<1 px", lambda r: r["motion"] < 1),
        ("1-2 px", lambda r: 1 <= r["motion"] < 2),
        ("2-4 px", lambda r: 2 <= r["motion"] < 4),
        (">=4 px", lambda r: r["motion"] >= 4),
    ]

    for name, fn in bins:

        vals = [
            r for r in results
            if fn(r)
        ]

        if not vals:
            continue

        print(f"\n{name}  N={len(vals)}")

        for key in ["raw", "diff", "oracle"]:

            values = np.array(
                [r[key] for r in vals],
                dtype=np.float32
            )

            print(
                f"  {key:8s} "
                f"mean={values.mean():.6f} "
                f"median={np.median(values):.6f}"
            )

    print()
    print("=" * 80)
    print("ORACLE GAIN")
    print("=" * 80)

    raw = np.array([r["raw"] for r in results])
    oracle = np.array([r["oracle"] for r in results])
    diff = np.array([r["diff"] for r in results])

    print(
        "Oracle/raw median ratio : "
        f"{np.median(oracle) / max(np.median(raw),1e-9):.3f}"
    )

    print(
        "Diff/raw median ratio   : "
        f"{np.median(diff) / max(np.median(raw),1e-9):.3f}"
    )


if __name__ == "__main__":
    main()
