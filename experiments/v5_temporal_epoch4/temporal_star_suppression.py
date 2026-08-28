import cv2
import numpy as np
from pathlib import Path

ROOT = Path.home() / "space-debris-ai"
IMG_DIR = ROOT / "data" / "v4" / "val" / "images"

files = sorted(
    IMG_DIR.glob("val_000000_frame_*.png")
)[:16]

frames = []

for f in files:
    img = cv2.imread(
        str(f),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(f"Cannot read {f}")

    frames.append(
        img.astype(np.float32) / 255.0
    )

print("=" * 80)
print("TEMPORAL STAR/BACKGROUND SUPPRESSION")
print("=" * 80)

for i in range(len(frames) - 1):

    a = frames[i]
    b = frames[i + 1]

    diff = np.abs(b - a)

    # Quantify how sparse the temporal residual is.
    print(
        f"{i:02d}->{i+1:02d} "
        f"mean={diff.mean():.6f} "
        f"p95={np.percentile(diff,95):.6f} "
        f"p99={np.percentile(diff,99):.6f} "
        f">0.01={np.mean(diff > 0.01)*100:.3f}% "
        f">0.02={np.mean(diff > 0.02)*100:.3f}% "
        f">0.03={np.mean(diff > 0.03)*100:.3f}%"
    )

print()
print("Interpretation:")
print("A small fraction of high-residual pixels is desirable.")
print("It means stationary background is mostly suppressed.")
