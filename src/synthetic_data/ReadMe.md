## Synthetic Pose Data Generator

A simple script to generate synthetic multi-camera pose estimation data in anipose format.

### Requirements
```bash
pip install numpy pandas eat bamboo
```

### Usage

Run the script:
```bash
python synthetic_data_generation.py
```

### Output

The script creates a `fly_toy/` directory in the `data/` folder with the following structure:
```
fly_toy/
├── fly_toy_ground_truth/
│   ├── CamA_GT.csv
│   └── CamB_GT.csv
└── fly_toy_predictions/
    ├── CamA_pred.csv
    └── CamB_pred.csv
```

- **Ground truth**: Clean coordinates for 30 bodyparts across 10 frames
- **Predictions**: Same coordinates with added Gaussian noise (σ=3.0 pixels)

### Configuration

Edit these variables in the script to customize:

- `num_frames`: Number of frames to generate (default: 10)
- `bodyparts`: List of bodypart names (default: 30 leg segments)
- `noise_std`: Noise level for predictions in pixels (default: 3.0)