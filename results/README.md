# Results

This directory is reserved for verified research artifacts. The exploratory notebook recorded the following values:

| Experiment | Metric | Value | Provenance |
| --- | --- | ---: | --- |
| Baseline development | Best validation Dice | 0.6195 | Later development checkpoint |
| Flip augmentation | Best validation Dice | 0.5205 | Separate random-start experiment |
| Interim test evaluation | Dice / IoU / loss | 0.5083 / 0.3753 / 0.9241 | Earlier checkpoint |

The interim test result must not be interpreted as final test performance for the later checkpoint with validation Dice 0.6195. No learning curve is included because the repository does not contain a standalone, verified export of the complete historical arrays. The original notebook remains the provenance source.

Run `python -m src.inference ...` to create a four-panel qualitative figure after supplying a checkpoint and input image.
