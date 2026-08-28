from pathlib import Path

import cv2
import numpy as np
import torch

from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset


class SpaceDebrisTemporalMedian1024Dataset(
    SpaceDebrisHeatmap1024Dataset
):
    """
    4-channel temporal representation:

        channel 0 = I(t)
        channel 1 = |I(t) - I(t-1)|
        channel 2 = |I(t+1) - I(t)|
        channel 3 = |I(t) - temporal_median|

    The target remains the current frame t.
    """

    def __init__(self, root, split="train", output_stride=4):
        super().__init__(
            root,
            split=split,
            output_stride=output_stride
        )

        self.lookup = {
            p.stem: p
            for p in self.images
        }

    def __getitem__(self, index):

        img_path = self.images[index]

        prefix, frame_number = img_path.name.rsplit(
            "_frame_", 1
        )

        frame_number = int(
            frame_number.split(".")[0]
        )

        sequence_stems = []

        for p in self.images:
            if p.name.startswith(prefix + "_frame_"):
                sequence_stems.append(p.stem)

        sequence_stems = sorted(sequence_stems)

        current_stem = img_path.stem

        try:
            current_idx = sequence_stems.index(current_stem)
        except ValueError as exc:
            raise RuntimeError(
                f"Cannot locate {current_stem} in sequence"
            ) from exc

        def read(path):
            img = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                raise RuntimeError(
                    f"Cannot read {path}"
                )

            return img.astype(np.float32) / 255.0

        current = read(img_path)

        if current_idx > 0:
            prev_path = self.lookup[
                sequence_stems[current_idx - 1]
            ]
            previous = read(prev_path)
        else:
            previous = current.copy()

        if current_idx + 1 < len(sequence_stems):
            next_path = self.lookup[
                sequence_stems[current_idx + 1]
            ]
            following = read(next_path)
        else:
            following = current.copy()

        # Use the whole sequence as a temporal background estimate.
        sequence_frames = []

        for stem in sequence_stems:
            sequence_frames.append(
                read(self.lookup[stem])
            )

        sequence_stack = np.stack(
            sequence_frames,
            axis=0
        )

        temporal_median = np.median(
            sequence_stack,
            axis=0
        ).astype(np.float32)

        diff_prev = np.abs(
            current - previous
        )

        diff_next = np.abs(
            following - current
        )

        diff_median = np.abs(
            current - temporal_median
        )

        temporal = np.stack(
            [
                current,
                diff_prev,
                diff_next,
                diff_median
            ],
            axis=0
        )

        base = super().__getitem__(index)

        base["frame"] = torch.from_numpy(
            temporal.astype(np.float32)
        )

        return base
