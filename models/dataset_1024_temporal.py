from pathlib import Path
import cv2
import numpy as np
import torch

from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset


class SpaceDebrisTemporal1024Dataset(SpaceDebrisHeatmap1024Dataset):
    """
    Returns:
        frame: 3 x 1024 x 1024

        channel 0 = I(t)
        channel 1 = |I(t) - I(t-1)|
        channel 2 = |I(t+1) - I(t)|

        Target corresponds to frame t.
    """

    def __init__(self, root, split="train", output_stride=4):
        super().__init__(
            root,
            split=split,
            output_stride=output_stride
        )

        self.stems = [
            p.stem for p in self.images
        ]

        self.lookup = {
            p.stem: p
            for p in self.images
        }

    def __getitem__(self, index):

        img_path = self.images[index]

        name = img_path.name

        if "_frame_" not in name:
            raise RuntimeError(
                f"Invalid sequence filename: {name}"
            )

        prefix, frame_number = name.rsplit(
            "_frame_", 1
        )

        frame_number = int(
            frame_number.split(".")[0]
        )

        prev_name = (
            f"{prefix}_frame_{frame_number - 1:03d}.png"
        )

        next_name = (
            f"{prefix}_frame_{frame_number + 1:03d}.png"
        )

        prev_path = self.lookup.get(
            Path(prev_name).stem
        )

        next_path = self.lookup.get(
            Path(next_name).stem
        )

        # Boundary handling:
        # repeat current frame when neighbor doesn't exist.
        if prev_path is None:
            prev_path = img_path

        if next_path is None:
            next_path = img_path

        def read(path):
            img = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                raise RuntimeError(
                    f"Cannot read {path}"
                )

            return (
                img.astype(np.float32) / 255.0
            )

        current = read(img_path)
        previous = read(prev_path)
        following = read(next_path)

        diff_prev = np.abs(
            current - previous
        )

        diff_next = np.abs(
            following - current
        )

        temporal = np.stack(
            [
                current,
                diff_prev,
                diff_next
            ],
            axis=0
        )

        # Get target/labels using the original dataset.
        base = super().__getitem__(index)

        base["frame"] = torch.from_numpy(
            temporal.astype(np.float32)
        )

        return base
