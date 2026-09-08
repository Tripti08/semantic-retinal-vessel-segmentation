import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from UNET.data import make_drive_datasets
from UNET.loss import DiceBCELoss, segmentation_metrics
from UNET.model import UNet
from UNET.utils import save_checkpoint, seed_everything


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_total = dice_total = iou_total = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss_total += criterion(logits, masks).item()
            dice, iou = segmentation_metrics(logits, masks)
            dice_total += dice
            iou_total += iou
    count = max(1, len(loader))
    return loss_total / count, dice_total / count, iou_total / count


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_set, val_set = make_drive_datasets(args.data_dir, (args.size, args.size), args.val_fraction)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_dice = -1.0
    print(f"Training on {device}: {len(train_set)} train, {len(val_set)} validation images")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, masks in progress:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_loss, dice, iou = evaluate(model, val_loader, criterion, device)
        print(f"train_loss={running_loss / max(1, len(train_loader)):.4f} val_loss={val_loss:.4f} dice={dice:.4f} iou={iou:.4f}")
        if dice > best_dice:
            best_dice = dice
            save_checkpoint(model, optimizer, epoch, dice, Path(args.checkpoint))
            print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/DRIVE")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
