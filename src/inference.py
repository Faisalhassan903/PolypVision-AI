import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .model import PolypUNet

def predict(model, path, device, threshold=0.5):
    original=Image.open(path).convert('RGB'); resized=original.resize((256,256),Image.Resampling.BILINEAR)
    arr=np.asarray(resized,dtype=np.float32)/255.; x=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
    return resized,prob,(prob>=threshold).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description='Create a research-only qualitative segmentation figure.')
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--output', default='results/prediction.png')
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = PolypUNet().to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    image, prob, mask = predict(model, a.image, device, a.threshold)
    arr = np.asarray(image)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(arr); ax[0].set_title('Original')
    ax[1].imshow(prob, cmap='viridis', vmin=0, vmax=1); ax[1].set_title('Probability Map')
    ax[2].imshow(mask, cmap='gray', vmin=0, vmax=1); ax[2].set_title('Predicted Mask')
    ax[3].imshow(arr); ax[3].imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.45); ax[3].set_title('Overlay')
    for axis in ax:
        axis.axis('off')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved', a.output)
if __name__=='__main__': main()
