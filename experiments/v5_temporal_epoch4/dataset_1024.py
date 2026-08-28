import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceDebrisHeatmap1024Dataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        output_stride=4
    ):
        self.root = Path(root)
        self.split = split
        self.output_stride = output_stride

        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"

        # Use EVERY frame, not only frame_000 from each sequence.
        self.images = sorted(
            self.image_dir.glob("*_frame_*.png")
        )

        self.image_size = 1024

        print(
            f"{split}: {len(self.images)} frames "
            f"(1024x1024)"
        )

    def __len__(self):
        return len(self.images)

    def _create_heatmap(
        self,
        labels,
        H,
        W
    ):

        out_h = H // self.output_stride
        out_w = W // self.output_stride

        heatmap = np.zeros(
            (1, out_h, out_w),
            dtype=np.float32
        )

        offsets = np.zeros(
            (2, out_h, out_w),
            dtype=np.float32
        )

        sizes = np.zeros(
            (2, out_h, out_w),
            dtype=np.float32
        )

        mask = np.zeros(
            (1, out_h, out_w),
            dtype=np.float32
        )

        for label in labels:

            # YOLO-like format:
            # class cx cy width height
            if len(label) < 5:
                continue

            cx = label[1] * W
            cy = label[2] * H

            bw = max(
                label[3] * W,
                1.0
            )

            bh = max(
                label[4] * H,
                1.0
            )

            cx_out = cx / self.output_stride
            cy_out = cy / self.output_stride

            cx_quant = int(round(cx_out))
            cy_quant = int(round(cy_out))

            if (
                cx_quant < 0
                or cx_quant >= out_w
                or cy_quant < 0
                or cy_quant >= out_h
            ):
                continue

            off_x = cx_out - cx_quant
            off_y = cy_out - cy_quant

            # Small Gaussian for tiny debris.
            sigma = max(
                2.0,
                max(bw, bh)
                / (2.0 * self.output_stride)
            )

            radius = max(
                1,
                int(sigma * 3)
            )

            for dy in range(
                -radius,
                radius + 1
            ):
                for dx in range(
                    -radius,
                    radius + 1
                ):

                    x = cx_quant + dx
                    y = cy_quant + dy

                    if (
                        0 <= x < out_w
                        and 0 <= y < out_h
                    ):

                        val = math.exp(
                            -(
                                dx * dx
                                + dy * dy
                            )
                            / (2.0 * sigma * sigma)
                        )

                        heatmap[
                            0,
                            y,
                            x
                        ] = max(
                            heatmap[
                                0,
                                y,
                                x
                            ],
                            val
                        )

            offsets[
                0,
                cy_quant,
                cx_quant
            ] = off_x

            offsets[
                1,
                cy_quant,
                cx_quant
            ] = off_y

            sizes[
                0,
                cy_quant,
                cx_quant
            ] = bw / self.output_stride

            sizes[
                1,
                cy_quant,
                cx_quant
            ] = bh / self.output_stride

            mask[
                0,
                cy_quant,
                cx_quant
            ] = 1.0

        return (
            heatmap,
            offsets,
            sizes,
            mask
        )

    def __getitem__(self, index):

        img_path = self.images[index]

        lab_path = (
            self.label_dir
            / f"{img_path.stem}.txt"
        )

        img = cv2.imread(
            str(img_path),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise RuntimeError(
                f"Cannot read {img_path}"
            )

        H, W = img.shape

        img = (
            img.astype(
                np.float32
            )
            / 255.0
        )

        img_t = (
            torch.from_numpy(img)
            .unsqueeze(0)
        )

        labels = []

        if lab_path.exists():

            text = (
                lab_path
                .read_text()
                .strip()
            )

            if text:

                for line in text.splitlines():

                    vals = list(
                        map(
                            float,
                            line.split()
                        )
                    )

                    if len(vals) >= 5:
                        labels.append(vals)

        hm, off, sz, msk = (
            self._create_heatmap(
                labels,
                H,
                W
            )
        )

        return {
            "frame": img_t,
            "heatmap": torch.from_numpy(hm),
            "offsets": torch.from_numpy(off),
            "sizes": torch.from_numpy(sz),
            "mask": torch.from_numpy(msk),
        }
