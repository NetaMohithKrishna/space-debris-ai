import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter

from models.heatmap_model_v5_temporal_median import HeatmapDetectorV5TemporalMedian
from models.dataset_1024_temporal_median import SpaceDebrisTemporalMedian1024Dataset


DEVICE = torch.device("cuda")
ROOT = Path.home() / "space-debris-ai" / "data" / "v4"


class CenterNetFocalLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=4.0, pos_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        pred = pred.clamp(1e-4, 1.0 - 1e-4)

        pos = target.eq(1.0).float()
        neg = target.lt(1.0).float()

        neg_weight = (1.0 - target) ** self.beta

        pos_loss = (
            -torch.log(pred)
            * ((1.0 - pred) ** self.alpha)
            * pos
            * self.pos_weight
        )

        neg_loss = (
            -torch.log(1.0 - pred)
            * (pred ** self.alpha)
            * neg_weight
            * neg
        )

        num_pos = pos.sum().clamp(min=1.0)

        return (
            pos_loss.sum() / num_pos
            + neg_loss.sum() / neg.sum().clamp(min=1.0)
        )


def peaks(hm, threshold=0.5):

    arr = hm.detach().cpu().numpy()

    local_max = maximum_filter(arr, size=5) == arr

    return np.argwhere(
        (arr >= threshold) & local_max
    )


def evaluate(model, loader):

    model.eval()

    total_gt = 0
    total_pred = 0
    total_tp = 0

    gt_scores = []

    with torch.no_grad():

        for batch in loader:

            frame = batch["frame"].to(DEVICE)
            gt = batch["heatmap"][0, 0].to(DEVICE)

            pred, _, _ = model(frame)

            pred = torch.sigmoid(pred[0, 0])

            gt_peaks = peaks(gt)
            pred_peaks = peaks(pred)

            total_gt += len(gt_peaks)
            total_pred += len(pred_peaks)

            matched = set()

            for gy, gx in gt_peaks:

                gt_scores.append(
                    pred[gy, gx].item()
                )

                best = None
                best_dist = float("inf")

                for j, (py, px) in enumerate(pred_peaks):

                    if j in matched:
                        continue

                    d = np.hypot(
                        px - gx,
                        py - gy
                    )

                    if d <= 5 and d < best_dist:
                        best = j
                        best_dist = d

                if best is not None:
                    matched.add(best)

            total_tp += len(matched)

    precision = total_tp / max(total_pred, 1)
    recall = total_tp / max(total_gt, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-9)
    )

    print(
        f"GT={total_gt} "
        f"Pred={total_pred} "
        f"TP={total_tp} "
        f"P={precision:.4f} "
        f"R={recall:.4f} "
        f"F1={f1:.4f}"
    )

    if gt_scores:
        print(
            f"GT score mean={np.mean(gt_scores):.4f} "
            f"max={np.max(gt_scores):.4f}"
        )


def main():

    dataset = SpaceDebrisTemporalMedian1024Dataset(
        ROOT,
        split="train",
        output_stride=4
    )

    subset = Subset(
        dataset,
        [0, 1, 2, 3, 4]
    )

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = HeatmapDetectorV5TemporalMedian().to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0
    )

    criterion = CenterNetFocalLoss(
        alpha=2.0,
        beta=4.0,
        pos_weight=5.0
    )

    print("=" * 70)
    print("V5-TEMPORAL 5-SAMPLE OVERFIT")
    print("=" * 70)

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f}M"
    )

    for epoch in range(1, 201):

        model.train()
        total_loss = 0.0

        for batch in loader:

            frame = batch["frame"].to(DEVICE)

            heatmap = batch["heatmap"].to(DEVICE)
            offsets = batch["offsets"].to(DEVICE)
            sizes = batch["sizes"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            pred_hm, pred_off, pred_size = model(frame)

            hm_loss = criterion(
                pred_hm.squeeze(1),
                heatmap.squeeze(1)
            )

            mask2 = mask.expand_as(pred_off)

            off_loss = nn.functional.l1_loss(
                pred_off * mask2,
                offsets * mask2,
                reduction="sum"
            ) / (mask2.sum() + 1e-6)

            size_loss = nn.functional.l1_loss(
                pred_size * mask2,
                sizes * mask2,
                reduction="sum"
            ) / (mask2.sum() + 1e-6)

            loss = (
                hm_loss
                + 0.5 * off_loss
                + 0.5 * size_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            total_loss += loss.item()

        if epoch == 1 or epoch % 20 == 0:

            print(
                f"\nEpoch {epoch:3d} "
                f"Loss={total_loss / len(loader):.6f}"
            )

            evaluate(
                model,
                loader
            )


if __name__ == "__main__":
    main()
