import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from models.heatmap_model_v4 import HeatmapDetectorV4
from models.dataset_1024 import SpaceDebrisHeatmap1024Dataset
from training.train_1024_improved import CenterNetFocalLoss


DEVICE = torch.device("cuda")
MODEL_PATH = "heatmap_detector_1024_improved_best.pth"

root = Path.home() / "space-debris-ai" / "data" / "v4"

criterion = CenterNetFocalLoss(alpha=2.0, beta=4.0)

model = HeatmapDetectorV4(
    input_channels=1,
    output_stride=4
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

def inspect_split(split):
    dataset = SpaceDebrisHeatmap1024Dataset(
        root,
        split=split,
        output_stride=4
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    model.eval()

    hm_losses = []
    off_losses = []
    size_losses = []
    totals = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            frames = batch["frame"].to(DEVICE)
            heatmaps = batch["heatmap"].to(DEVICE)
            offsets = batch["offsets"].to(DEVICE)
            sizes = batch["sizes"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            pred_heatmap, pred_offsets, pred_sizes = model(frames)

            hm_loss = criterion(
                pred_heatmap.squeeze(1),
                heatmaps.squeeze(1)
            )

            mask_exp = masks.expand_as(pred_offsets)

            off_loss = nn.functional.l1_loss(
                pred_offsets * mask_exp,
                offsets * mask_exp,
                reduction="sum"
            ) / (mask_exp.sum() + 1e-6)

            sz_loss = nn.functional.l1_loss(
                pred_sizes * mask_exp,
                sizes * mask_exp,
                reduction="sum"
            ) / (mask_exp.sum() + 1e-6)

            total = hm_loss + 0.5 * off_loss + 0.5 * sz_loss

            hm_losses.append(hm_loss.item())
            off_losses.append(off_loss.item())
            size_losses.append(sz_loss.item())
            totals.append(total.item())

            if i < 10:
                print(
                    f"{split} sample {i:02d}: "
                    f"HM={hm_loss.item():.4f} "
                    f"OFF={off_loss.item():.4f} "
                    f"SIZE={sz_loss.item():.4f} "
                    f"TOTAL={total.item():.4f}"
                )

    print(f"\n{split} summary")
    print(f"Mean HM   : {sum(hm_losses)/len(hm_losses):.6f}")
    print(f"Mean OFF  : {sum(off_losses)/len(off_losses):.6f}")
    print(f"Mean SIZE : {sum(size_losses)/len(size_losses):.6f}")
    print(f"Mean TOTAL: {sum(totals)/len(totals):.6f}")
    print(f"Max TOTAL : {max(totals):.6f}")


print("=" * 70)
inspect_split("train")

print("\n" + "=" * 70)
inspect_split("val")
