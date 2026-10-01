# Model Checkpoints

The trained checkpoint is intentionally not included in this repository. The notebook recorded a 30 MB checkpoint, so normal GitHub storage is unsuitable. Keep it in Git LFS or, preferably for a research artifact, attach it to a versioned GitHub Release or another institutional/model registry.

Copy a supplied checkpoint into this directory before evaluation or inference:

```text
models/polypvision_unet_baseline_best.pth
```

Weights are not modified by the repository scripts. Training refuses to overwrite an existing output path.
