import torch

@torch.no_grad()
def per_image_metrics(logits, targets, threshold=0.5, eps=1e-6):
    """Return per-image Dice and IoU tensors; average over images, not batches."""
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = targets.float().flatten(1)
    inter = (pred*target).sum(1)
    p, t = pred.sum(1), target.sum(1)
    dice = (2*inter+eps)/(p+t+eps)
    iou = (inter+eps)/(p+t-inter+eps)
    return dice, iou
