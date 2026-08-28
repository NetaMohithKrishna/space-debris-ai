import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader

from models.dataset_1024_temporal import SpaceDebrisTemporal1024Dataset
from models.heatmap_model_v5_temporal import HeatmapDetectorV5Temporal


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda")
ROOT = Path.home() / "space-debris-ai" / "data" / "v4"

THRESHOLD = 0.50
NMS_RADIUS = 5
MATCH_RADIUS = 5


class WeightedCenterNetFocalLoss(nn.Module):
    """
    Separate normalization of positive/negative terms, with
    a moderate negative-loss multiplier.

    loss = positive_loss + NEG_WEIGHT * negative_loss
    """

    def __init__(
        self,
        alpha=2.0,
        beta=4.0,
        pos_weight=1.0,
        neg_weight=10.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        pred = pred.clamp(1e-4, 1.0 - 1e-4)

        pos = target.eq(1.0).float()
        neg = target.lt(1.0).float()

        neg_weights = (1.0 - target) ** self.beta

        pos_loss = (
            -torch.log(pred)
            * ((1.0 - pred) ** self.alpha)
            * pos
            * self.pos_weight
        )

        neg_loss = (
            -torch.log(1.0 - pred)
            * (pred ** self.alpha)
            * neg_weights
            * neg
        )

        num_pos = pos.sum().clamp(min=1.0)
        num_neg = neg.sum().clamp(min=1.0)

        pos_loss = pos_loss.sum() / num_pos
        neg_loss = neg_loss.sum() / num_neg

        return pos_loss + self.neg_weight * neg_loss


def get_peaks(heatmap, threshold=THRESHOLD):
    local_max = maximum_filter(
        heatmap,
        size=5
    ) == heatmap

    coords = np.argwhere(
        (heatmap >= threshold) & local_max
    )

    coords = sorted(
        coords,
        key=lambda p: heatmap[p[0], p[1]],
        reverse=True
    )

    selected = []

    for y, x in coords:

        keep = True

        for sy, sx in selected:

            if (
                (x - sx) ** 2
                + (y - sy) ** 2
                <= NMS_RADIUS ** 2
            ):
                keep = False
                break

        if keep:
            selected.append((y, x))

    return selected


def get_gt_centers(gt):

    local_max = maximum_filter(
        gt,
        size=5
    ) == gt

    coords = np.argwhere(
        (gt >= 0.5) & local_max
    )

    return [
        (int(y), int(x))
        for y, x in coords
    ]


def match_predictions(preds, gts):

    matched_gt = set()
    tp = 0

    for py, px in preds:

        best = None
        best_dist = float("inf")

        for i, (gy, gx) in enumerate(gts):

            if i in matched_gt:
                continue

            dist = np.hypot(
                px - gx,
                py - gy
            )

            if (
                dist <= MATCH_RADIUS
                and dist < best_dist
            ):
                best = i
                best_dist = dist

        if best is not None:
            matched_gt.add(best)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp

    return tp, fp, fn


def calculate_loss(
    model,
    batch,
    criterion,
    device
):

    frames = batch["frame"].to(
        device,
        non_blocking=True
    )

    heatmaps = batch["heatmap"].to(
        device,
        non_blocking=True
    )

    offsets = batch["offsets"].to(
        device,
        non_blocking=True
    )

    sizes = batch["sizes"].to(
        device,
        non_blocking=True
    )

    masks = batch["mask"].to(
        device,
        non_blocking=True
    )

    pred_heatmap, pred_offsets, pred_sizes = model(
        frames
    )

    heatmap_loss = criterion(
        pred_heatmap.squeeze(1),
        heatmaps.squeeze(1)
    )

    mask_exp = masks.expand_as(pred_offsets)

    offset_loss = nn.functional.l1_loss(
        pred_offsets * mask_exp,
        offsets * mask_exp,
        reduction="sum"
    ) / (
        mask_exp.sum() + 1e-6
    )

    size_loss = nn.functional.l1_loss(
        pred_sizes * mask_exp,
        sizes * mask_exp,
        reduction="sum"
    ) / (
        mask_exp.sum() + 1e-6
    )

    total = (
        heatmap_loss
        + 0.5 * offset_loss
        + 0.5 * size_loss
    )

    return total


def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for batch in loader:

            loss = calculate_loss(
                model,
                batch,
                criterion,
                device
            )

            total_loss += loss.item()

            frames = batch["frame"].to(
                device,
                non_blocking=True
            )

            gt_batch = batch["heatmap"]

            pred, _, _ = model(frames)

            pred = torch.sigmoid(
                pred[:, 0]
            ).cpu().numpy()

            gt_batch = gt_batch[:, 0].numpy()

            for b in range(pred.shape[0]):

                predictions = get_peaks(
                    pred[b]
                )

                gts = get_gt_centers(
                    gt_batch[b]
                )

                tp, fp, fn = match_predictions(
                    predictions,
                    gts
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

    mean_loss = (
        total_loss / max(len(loader), 1)
    )

    precision = (
        total_tp
        / max(total_tp + total_fp, 1)
    )

    recall = (
        total_tp
        / max(total_tp + total_fn, 1)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-9)
    )

    return (
        mean_loss,
        total_tp,
        total_fp,
        total_fn,
        precision,
        recall,
        f1
    )


def main():

    print(
        "Training V5-TEMPORAL on CUDA "
        "at 1024x1024"
    )

    print(
        f"Input channels: 3 "
        "(current + previous diff + next diff)"
    )

    print(
        f"Negative loss weight: 10.0"
    )

    train_dataset = SpaceDebrisTemporal1024Dataset(
        ROOT,
        split="train",
        output_stride=4
    )

    val_dataset = SpaceDebrisTemporal1024Dataset(
        ROOT,
        split="val",
        output_stride=4
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = HeatmapDetectorV5Temporal(
        output_stride=4
    ).to(DEVICE)

    criterion = WeightedCenterNetFocalLoss(
        alpha=2.0,
        beta=4.0,
        pos_weight=1.0,
        neg_weight=10.0
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{params:,} ({params / 1e6:.3f}M)"
    )

    print("-" * 80)

    num_epochs = 20

    best_f1 = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):

        start = time.time()

        model.train()

        train_total = 0.0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = calculate_loss(
                model,
                batch,
                criterion,
                DEVICE
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            train_total += loss.item()

            if (batch_idx + 1) % 250 == 0:

                speed = (
                    (batch_idx + 1)
                    / max(time.time() - start, 1e-6)
                )

                print(
                    f"  Epoch {epoch} "
                    f"Batch {batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"Loss={loss.item():.4f} "
                    f"Speed={speed:.2f}/s"
                )

        train_loss = (
            train_total
            / max(len(train_loader), 1)
        )

        (
            val_loss,
            tp,
            fp,
            fn,
            precision,
            recall,
            f1
        ) = evaluate_model(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        scheduler.step(val_loss)

        elapsed = time.time() - start

        print()
        print("=" * 80)
        print(
            f"Epoch {epoch}/{num_epochs}"
        )
        print("=" * 80)

        print(
            f"Train Loss : {train_loss:.6f}"
        )

        print(
            f"Val Loss   : {val_loss:.6f}"
        )

        print(
            f"TP         : {tp}"
        )

        print(
            f"FP         : {fp}"
        )

        print(
            f"FN         : {fn}"
        )

        print(
            f"Precision  : {precision:.6f}"
        )

        print(
            f"Recall     : {recall:.6f}"
        )

        print(
            f"F1         : {f1:.6f}"
        )

        print(
            f"Time       : {elapsed:.1f}s"
        )

        print(
            f"LR         : "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            f"checkpoint_v5_temporal_epoch_{epoch:02d}.pth"
        )

        if (
            f1 > best_f1
            or (
                f1 == best_f1
                and val_loss < best_val_loss
            )
        ):
            best_f1 = f1
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "heatmap_detector_v5_temporal_best.pth"
            )

            print(
                "-> Saved best temporal model"
            )

        print()

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
