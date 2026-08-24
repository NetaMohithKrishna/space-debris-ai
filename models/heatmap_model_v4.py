import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        return self.conv(x)

class HeatmapDetectorV4(nn.Module):
    def __init__(self, input_channels=1, output_stride=4):
        super().__init__()
        self.output_stride = output_stride
        # Encoder: input 512 -> 128 (stride 4)
        self.enc1 = DoubleConv(input_channels, 32)
        self.pool1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = DoubleConv(64, 128)
        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(128, 1, kernel_size=1)

        # Low initial prior for sparse object-center heatmaps.
        # sigmoid(-2.19) ≈ 0.10
        nn.init.constant_(self.heatmap_head.bias, -2.19)
        self.offset_head = nn.Conv2d(128, 2, kernel_size=1)
        self.size_head = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.refine(x)
        heatmap = self.heatmap_head(x)
        offsets = self.offset_head(x)
        sizes = self.size_head(x)
        return heatmap, offsets, sizes
