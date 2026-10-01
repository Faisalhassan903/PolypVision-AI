from pathlib import Path
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def build_pairs(root):
    root = Path(root)
    image_dir, mask_dir = root/'images', root/'masks'
    duplicate_suffix = re.compile(r' \(\d+\)$')
    images = {
        p.stem: p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not duplicate_suffix.search(p.stem)
    }
    masks = {
        p.stem: p for p in mask_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not duplicate_suffix.search(p.stem)
    }
    ids = sorted(images.keys() & masks.keys())
    return ids, images, masks

def split_ids(ids, seed=42, n_train=700, n_val=150):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]

class PairedFlip:
    def __init__(self, hflip_p=0.5, vflip_p=0.5):
        self.hflip_p, self.vflip_p = hflip_p, vflip_p
    def __call__(self, image, mask):
        if random.random() < self.hflip_p:
            image, mask = torch.flip(image,[2]), torch.flip(mask,[2])
        if random.random() < self.vflip_p:
            image, mask = torch.flip(image,[1]), torch.flip(mask,[1])
        return image, mask

class PolypDataset(Dataset):
    def __init__(self, ids, images, masks, indices, image_size=256, transform=None):
        self.sample_ids = [ids[int(i)] for i in indices]
        self.images, self.masks = images, masks
        self.image_size, self.transform = image_size, transform
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        size = (self.image_size, self.image_size)
        image = Image.open(self.images[sid]).convert('RGB').resize(size, Image.Resampling.BILINEAR)
        mask = Image.open(self.masks[sid]).convert('L').resize(size, Image.Resampling.NEAREST)
        image = torch.from_numpy(np.asarray(image,dtype=np.float32)/255.0).permute(2,0,1)
        mask = torch.from_numpy((np.asarray(mask)>127).astype(np.float32)).unsqueeze(0)
        if self.transform: image, mask = self.transform(image, mask)
        return image, mask
