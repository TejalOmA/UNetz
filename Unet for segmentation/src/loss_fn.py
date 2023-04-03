import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):

        preds = torch.sigmoid(preds)

        # Flatten predictions and label tensors
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)

        intersection = torch.sum(preds_flat * labels_flat)
        union = torch.sum(preds_flat) + torch.sum(labels_flat)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1-dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)

        # Flatten predictions and label tensors
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)

        intersection = torch.sum(preds_flat * labels_flat)
        union = torch.sum(preds_flat) + torch.sum(labels_flat)

        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        bce = F.binary_cross_entropy(preds_flat, labels_flat, reduction='mean')

        dice_bce = dice_loss + bce

        return dice_bce

