# PolypVision AI

PolypVision AI is a from-scratch PyTorch baseline for binary colonoscopy polyp segmentation on Kvasir-SEG. It is a research and learning prototype, not a clinical diagnostic system.

## Overview

The project implements a custom U-Net-style encoder-decoder with skip connections, trained from random initialization. It exists to make dataset checks, preprocessing, optimization, evaluation, and inference understandable and reproducible.

## Dataset

Download Kvasir-SEG from its [official source](https://datasets.simula.no/kvasir-seg/) and do not commit the dataset here. The expected layout is:

```text
Kvasir-SEG/
├── images/
└── masks/
```

The working copy initially contained duplicate filename artifacts such as `(1)` copies. Canonical image-mask pairing excludes numbered duplicate suffixes and produced 1,000 valid pairs.

## Data Split

| Split | Images |
| --- | ---: |
| Train | 700 |
| Validation | 150 |
| Test | 150 |
| Random seed | 42 |

Images are converted to RGB, resized to 256 x 256 with bilinear interpolation, and scaled to `[0, 1]`. Masks are converted to grayscale, resized with nearest-neighbor interpolation, thresholded at `>127`, and returned as `[1, H, W]` tensors.

## Architecture

The custom `PolypUNet` uses convolution + ReLU blocks, max pooling, transposed-convolution upsampling, and skip concatenation. It has no BatchNorm, pretrained encoder, attention module, or torchvision/MONAI segmentation backbone.

```text
Trainable parameters: 7,760,097
Input: 3 x 256 x 256
Output: 1 x 256 x 256 logits
```

## Objective and Metrics

Training minimizes `BCEWithLogitsLoss + soft Dice loss` with Adam and an initial learning rate of `0.001`.

For metrics, logits are passed through sigmoid and thresholded at `0.5`. Dice and IoU are computed separately for each image and then averaged over images, so batches do not receive unequal weight because of a smaller final batch:

```text
Dice = (2 * intersection + eps) / (predicted pixels + target pixels + eps)
IoU  = (intersection + eps) / (union + eps)
```

## Experiments and Results

These values are historical notebook results and are not recomputed or combined across checkpoints.

| Experiment | Supported result |
| --- | ---: |
| Baseline development: best later validation Dice | **0.6195** |
| Flip augmentation: best validation Dice | **0.5205** |
| Earlier interim test evaluation: Dice | **0.5083** |
| Earlier interim test evaluation: IoU | **0.3753** |
| Earlier interim test evaluation: loss | **0.9241** |

The interim test metrics belong to an earlier checkpoint evaluated before later development continued. They must not be presented as test performance for the later checkpoint that reached validation Dice 0.6195. The simple flip augmentation experiment did not improve validation Dice. See [results/README.md](results/README.md) and the notebooks for provenance.

## Inference

Create a qualitative figure containing Original, Probability Map, Predicted Mask, and Overlay panels:

```bash
python -m src.inference \
    --image path/to/image.jpg \
    --checkpoint models/polypvision_unet_baseline_best.pth \
    --output results/prediction.png
```

This produces a predicted segmentation for research inspection only. It is not a clinical diagnosis.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train without automatic retraining during setup:

```bash
python -m src.train --data /path/to/Kvasir-SEG --epochs 35 --output models/best.pth
```

Training uses seed 42 by default, selects CUDA when available, saves the best validation checkpoint, and refuses to overwrite an existing checkpoint path. Use explicit paths and a specified checkpoint for evaluation:

```bash
python -m src.evaluate \
    --data /path/to/Kvasir-SEG \
    --checkpoint models/best.pth \
    --split val
```

The original experimental notebook is retained in `notebooks/PolypVision_original.ipynb`, including historical Colab paths, for provenance. The cleaned notebook documents the experiments but is not a replacement for a preregistered evaluation protocol.

## Limitations and Future Work

This is a small, single-dataset development study with one random split, no external-dataset validation, and no clinical validation. The test split was inspected during development, so the interim test result is not an untouched final estimate. The model is not for diagnosis or patient-care decisions.

Future work includes a clean predefined evaluation protocol, external validation, architecture comparisons, domain-appropriate augmentation, semi-supervised segmentation, registration-guided pseudo-labeling, and comparison with established polyp-segmentation architectures. ColonSegNet is mentioned only as related/future comparative research; this project does not reproduce it.

## Repository Structure

```text
src/        Reusable dataset, model, loss, metric, training, evaluation, and inference code
notebooks/  Cleaned experiments and preserved original provenance
models/     Checkpoint instructions; weights are not committed
results/    Verified result notes and generated figures
```

## References

- Jha, D. et al. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. MultiMedia Modeling, 451-462. https://doi.org/10.1007/978-3-030-37734-2_37
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597
- Jha, D. et al. (2021). *Real-Time Polyp Detection, Localization and Segmentation in Colonoscopy Using Deep Learning*. IEEE Access, 9, 40496-40510. https://github.com/DebeshJha/ColonSegNet
