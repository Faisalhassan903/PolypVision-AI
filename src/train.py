import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PairedFlip, PolypDataset, build_pairs, split_ids
from .losses import bce_dice_loss
from .metrics import per_image_metrics
from .model import PolypUNet

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, n_batches, dice_all, iou_all = 0.0, 0, [], []
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        d, i = per_image_metrics(logits.detach(), y)
        dice_all.append(d.cpu()); iou_all.append(i.cpu())
        loss_sum += loss.item(); n_batches += 1
    return loss_sum / n_batches, torch.cat(dice_all).mean().item(), torch.cat(iou_all).mean().item()

def main():
    ap = argparse.ArgumentParser(description='Train the PolypVision U-Net baseline.')
    ap.add_argument('--data', required=True, help='Kvasir-SEG root containing images/ and masks/')
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--output', default='models/best.pth')
    a = ap.parse_args()
    output_path = Path(a.output)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {output_path}')

    seed_all(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids, images, masks = build_pairs(a.data)
    if len(ids) != 1000:
        print(f'Found {len(ids)} valid image-mask pairs.')
    tr, va, _ = split_ids(ids, a.seed)
    transform = PairedFlip() if a.augment else None
    train = PolypDataset(ids, images, masks, tr, transform=transform)
    val = PolypDataset(ids, images, masks, va)
    loader_generator = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val, batch_size=a.batch_size)
    model = PolypUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1.0
    best_state = None
    print(f'Device: {device} | Train: {len(train)} | Validation: {len(val)}')
    for epoch in range(1,a.epochs+1):
        tl,td,ti=run_epoch(model,train_loader,device,opt)
        vl,vd,vi=run_epoch(model,val_loader,device)
        print(f'Epoch {epoch:02d} | train loss {tl:.4f} dice {td:.4f} | val loss {vl:.4f} dice {vd:.4f} iou {vi:.4f}')
        if vd > best:
            best = vd
            best_state = copy.deepcopy(model.state_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict':best_state, 'best_val_dice':best, 'seed':a.seed, 'image_size':256}, output_path)
    print('Saved', output_path, 'best validation Dice', f'{best:.4f}')

if __name__=='__main__': main()
