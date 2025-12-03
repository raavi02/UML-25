# Visualization Script

## Running the Visualization Script

The script takes the following arguments:

* `--video` : Path to input `.mp4` file
* `--dataset` : `fly` or `chickadee`
* `--model` : `MLP` or `GNN`
* `--seed` : Integer (e.g., 42, 7, 123)
* `--ood` : Use `_new` suffix files (default = True)

## Example Commands

**Fly dataset, MLP model, seed 42**
```bash
python src/visualize/visualize_keypoints.py \
    --video /path/to/Cam-A_video1.mp4 \
    --dataset fly \
    --model MLP \
    --seed 42
```

**Fly dataset, GNN model, seed 7**
```bash
python src/visualize/visualize_keypoints.py \
    --video /path/to/Cam-C_video3.mp4 \
    --dataset fly \
    --model GNN \
    --seed 7
```

**Disable OOD mode** (use non-`_new` csv files)
```bash
python src/visualize/visualize_keypoints.py \
    --video /path/to/Cam-B_seq5.mp4 \
    --dataset chickadee \
    --model MLP \
    --seed 1 \
    --ood False
```

## Output

Processed visualization is saved to:
```
outputs/visualizations/<MODEL>_<SEED>/<video_name>_overlay.mp4
```

Example:
```
outputs/visualizations/MLP_42/Cam-A_test1_overlay.mp4
```

## Notes

* The script automatically detects the camera from the filename (e.g., `Cam-A`).
* It draws:
   * Ground Truth (green)
   * Original Predictions (red)
   * Refined Predictions (yellow)
* A color legend is displayed in the top-right corner.