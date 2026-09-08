import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from UNET.data import DriveDataset
from UNET.loss import segmentation_metrics
from UNET.model import UNet
from UNET.utils import save_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    test_root = Path(args.data_dir) / "test"
    mask_dir = test_root / "1st_manual"
    if not mask_dir.exists():
        raise FileNotFoundError("Expected test masks at data/DRIVE/test/1st_manual")
    dataset = DriveDataset(test_root / "images", mask_dir, image_size=(args.size, args.size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = UNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dice_total = iou_total = 0.0
    output_dir = Path(args.output_dir)
    with torch.no_grad():
        for index, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            dice, iou = segmentation_metrics(logits, masks)
            dice_total += dice
            iou_total += iou
            prediction = (torch.sigmoid(logits) > 0.5).float()
            save_mask(prediction[0], output_dir / f"prediction_{index:03d}.png")
    count = max(1, len(loader))
    print(f"Test Dice: {dice_total / count:.4f}")
    print(f"Test IoU:   {iou_total / count:.4f}")
    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/DRIVE")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output_dir", default="results/predictions")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
