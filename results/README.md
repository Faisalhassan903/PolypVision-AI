# Results

Generated from the local Kvasir-SEG copy, seed `42`, and `models/polypvision_unet_baseline_best.pth`:

- `qualitative_results.png`: two highest-Dice and two lowest-Dice validation examples, each shown as Input Image, Ground Truth, Probability Map, Predicted Mask, and Overlay.
- `test_failure_cases.png`: exploratory analysis of two lowest per-image Dice test examples. The test split had already been inspected during development, so this is not an untouched final test evaluation.
- `polypunet_architecture.png`: the verified from-scratch PolypUNet channel progression and skip connections.

The validation examples are selected by per-image Dice, not manually cherry-picked. No training curve is included because the notebook does not preserve complete, standalone loss history for the later development phase.
