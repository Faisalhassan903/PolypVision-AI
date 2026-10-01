import torch
import torch.nn as nn

_bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(1)
    dice = (2*intersection + eps)/(probs.sum(1)+targets.sum(1)+eps)
    return 1-dice.mean()

def bce_dice_loss(logits, targets):
    return _bce(logits, targets) + dice_loss(logits, targets)
