# multiview_gnn_refiner

Lightweight GNN module to refine 2-D keypoints by enforcing skeleton + cross-camera consistency.

## Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import sys; sys.path.append('src'); print('OK')"
```

## Dataset Folder Layout

Each dataset lives under `./data/{dataset_name}/`.
For example, the **fly** dataset is organized as:

```bash
data/
└── fly/
    ├── fly_ground_truth/
    ├── fly_ground_truth_OOD/
    ├── fly_predictions/
    └── fly_predictions_OOD/
```
These can simply be copy-pasted from the shared Drive

## Running the GNN Refiner
```bash
python python .\src\model\gnn_refiner.py --config .\configs\pipeline.yaml
```
