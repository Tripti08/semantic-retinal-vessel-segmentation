import torch
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return bce + (1 - dice.mean())


@torch.no_grad()
def segmentation_metrics(logits, targets, threshold=0.5, smooth=1.0):
    predictions = (torch.sigmoid(logits) > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    dice = ((2 * intersection + smooth) / (predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth)).mean()
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return dice.item(), iou.item()
