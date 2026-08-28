import cv2
import numpy as np
from pathlib import Path

ROOT = Path.home() / "space-debris-ai"
IMG_DIR = ROOT / "data" / "v4" / "val" / "images"

files = sorted(
    IMG_DIR.glob("val_000000_frame_*.png")
)

print("=" * 80)
print("TEMPORAL DIFFERENCE DIAGNOSTIC")
print("=" * 80)

print(f"Frames found: {len(files)}")

if len(files) < 2:
    raise RuntimeError("Need at least two consecutive frames.")

frames = []

for f in files[:16]:

    img = cv2.imread(
        str(f),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(f"Cannot read {f}")

    frames.append(
        img.astype(np.float32) / 255.0
    )

for i in range(len(frames) - 1):

    a = frames[i]
    b = frames[i + 1]

    diff = np.abs(b - a)

    print(
        f"{i:02d}->{i+1:02d}: "
        f"mean={diff.mean():.6f}, "
        f"median={np.median(diff):.6f}, "
        f"p95={np.percentile(diff,95):.6f}, "
        f"p99={np.percentile(diff,99):.6f}, "
        f"max={diff.max():.6f}"
    )
