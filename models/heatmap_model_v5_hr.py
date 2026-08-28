import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class HeatmapDetectorV5HR(nn.Module):
    """
    Lightweight high-resolution version of V4.

    Input:
        1 x 1024 x 1024

    Output:
        heatmap  1 x 256 x 256
        offsets  2 x 256 x 256
        sizes    2 x 256 x 256
    """

    def __init__(self, input_channels=1, output_stride=4):
        super().__init__()

        self.output_stride = output_stride

        # -----------------------------
        # Encoder
        # -----------------------------
        self.enc1 = DoubleConv(
            input_channels,
            32
        )

        self.pool1 = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        # -----------------------------
        # High-resolution skip
        # 512x512x64
        #       ->
        # 256x256x32
        # -----------------------------
        self.hr_project = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        # -----------------------------
        # Main branch
        # -----------------------------
        self.pool2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        # 128 main + 32 high-resolution
        self.fusion = nn.Sequential(
            nn.Conv2d(
                160,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        # -----------------------------
        # Detection heads
        # -----------------------------
        self.heatmap_head = nn.Conv2d(
            128,
            1,
            kernel_size=1
        )

        self.offset_head = nn.Conv2d(
            128,
            2,
            kernel_size=1
        )

        self.size_head = nn.Conv2d(
            128,
            2,
            kernel_size=1
        )

        # Same sparse-object prior as V4
        nn.init.constant_(
            self.heatmap_head.bias,
            -2.19
        )

    def forward(self, x):

        # 1024x1024
        x = self.enc1(x)

        # 512x512
        x = self.pool1(x)

        # 512x512x64
        x = self.enc2(x)

        # Save high-resolution information
        hr = self.hr_project(x)

        # Main branch -> 256x256
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.refine(x)

        # Fuse
        x = torch.cat(
            [x, hr],
            dim=1
        )

        x = self.fusion(x)

        heatmap = self.heatmap_head(x)
        offsets = self.offset_head(x)
        sizes = self.size_head(x)

        return heatmap, offsets, sizes
