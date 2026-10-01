import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import PolypDataset, build_pairs, split_ids
from .losses import bce_dice_loss
from .metrics import per_image_metrics
from .model import PolypUNet

def main():
    ap = argparse.ArgumentParser(description='Evaluate an explicit PolypVision checkpoint.')
    ap.add_argument('--data', required=True, help='Kvasir-SEG root containing images/ and masks/')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids, images, masks = build_pairs(a.data)
    _, val_indices, test_indices = split_ids(ids, a.seed)
    indices = val_indices if a.split == 'val' else test_indices
    loader = DataLoader(PolypDataset(ids, images, masks, indices), batch_size=8)
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = PolypUNet().to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    losses, dices, ious = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            losses.append(bce_dice_loss(logits, y.to(device)).item())
            dice, iou = per_image_metrics(logits, y.to(device))
            dices.append(dice.cpu())
            ious.append(iou.cpu())
    print(f'Split: {a.split} ({len(indices)} images)')
    print(f'Mean loss: {sum(losses) / len(losses):.4f}')
    print(f'Mean per-image Dice: {torch.cat(dices).mean().item():.4f}')
    print(f'Mean per-image IoU: {torch.cat(ious).mean().item():.4f}')
if __name__=='__main__': main()
