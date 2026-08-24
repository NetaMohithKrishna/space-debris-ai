import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np
import time
import gc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from models.heatmap_model_v4 import HeatmapDetectorV4
from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset

class CenterNetFocalLoss(nn.Module):
    """
    CenterNet-style modified focal loss for heatmap prediction.

    Positive locations are target == 1.
    All other locations are negatives, weighted by (1-target)^beta.
    The total loss is normalized by the number of positive centers.
    """
    def __init__(self, alpha=2.0, beta=4.0, pos_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.clamp(min=1e-4, max=1.0 - 1e-4)

        pos_inds = target.eq(1.0).float()
        neg_inds = target.lt(1.0).float()

        neg_weights = (1.0 - target) ** self.beta

        pos_loss = (
            -torch.log(pred)
            * ((1.0 - pred) ** self.alpha)
            * pos_inds * self.pos_weight
        )

        neg_loss = (
            -torch.log(1.0 - pred)
            * (pred ** self.alpha)
            * neg_weights
            * neg_inds
        )

        num_pos = pos_inds.sum()

        if num_pos > 0:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            loss = neg_loss.sum()

        return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    start = time.time()
    for batch_idx, batch in enumerate(loader):
        frames = batch["frame"].to(device, non_blocking=True)
        heatmaps = batch["heatmap"].to(device, non_blocking=True)
        offsets = batch["offsets"].to(device, non_blocking=True)
        sizes = batch["sizes"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

#        # Random horizontal flip augmentation
#        # Image/heatmap/offsets/sizes/mask must undergo the same spatial transform.
#        if random.random() < 0.5:
#            frames = torch.flip(frames, dims=[3])
#            heatmaps = torch.flip(heatmaps, dims=[3])
#            offsets = torch.flip(offsets, dims=[3])
#            offsets[:, 0] *= -1
#            sizes = torch.flip(sizes, dims=[3])
#            masks = torch.flip(masks, dims=[3])

        optimizer.zero_grad()
        pred_heatmap, pred_offsets, pred_sizes = model(frames)

        heatmap_loss = criterion(pred_heatmap.squeeze(1), heatmaps.squeeze(1))
        mask_exp = masks.expand_as(pred_offsets)
        offset_loss = nn.functional.l1_loss(pred_offsets * mask_exp, offsets * mask_exp, reduction='sum') / (mask_exp.sum() + 1e-6)
        size_loss = nn.functional.l1_loss(pred_sizes * mask_exp, sizes * mask_exp, reduction='sum') / (mask_exp.sum() + 1e-6)
        loss = heatmap_loss + 0.5 * offset_loss + 0.5 * size_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 25 == 0:
            elapsed = time.time() - start
            speed = (batch_idx + 1) / elapsed
            print(f"  Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}, Speed: {speed:.2f} batches/s")

    return total_loss / len(loader)

def main():
    device = torch.device("cuda")
    print(f"Training on {device} at 1024×1024 with improved loss")

    data_root = Path.home() / "space-debris-ai" / "data" / "v4"
    train_dataset = SpaceDebrisHeatmap1024Dataset(data_root, split="train", output_stride=4)
    val_dataset = SpaceDebrisHeatmap1024Dataset(data_root, split="val", output_stride=4)

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

    model = HeatmapDetectorV4(input_channels=1, output_stride=4).to(device)
    criterion = CenterNetFocalLoss(alpha=2.0, beta=4.0, pos_weight=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)

    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("Using standard CenterNet focal loss (no positive weight)")
    print("-" * 80)

    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frame"].to(device, non_blocking=True)
                heatmaps = batch["heatmap"].to(device, non_blocking=True)
                offsets = batch["offsets"].to(device, non_blocking=True)
                sizes = batch["sizes"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

                pred_heatmap, pred_offsets, pred_sizes = model(frames)
                heatmap_loss = criterion(pred_heatmap.squeeze(1), heatmaps.squeeze(1))
                mask_exp = masks.expand_as(pred_offsets)
                offset_loss = nn.functional.l1_loss(pred_offsets * mask_exp, offsets * mask_exp, reduction='sum') / (mask_exp.sum() + 1e-6)
                size_loss = nn.functional.l1_loss(pred_sizes * mask_exp, sizes * mask_exp, reduction='sum') / (mask_exp.sum() + 1e-6)
                loss = heatmap_loss + 0.5 * offset_loss + 0.5 * size_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_time = time.time() - start
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "heatmap_detector_1024_improved_best.pth")
            print(f"  -> Saved best model")

        gc.collect()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
