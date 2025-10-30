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

## GPU Quickstart Guide
| Item              | Value                                                |
| ----------------- | ---------------------------------------------------- |
| **Project**       | `uml-gnn-transfer`                                   |
| **Instance name** | `gnn-gpu-1`                                          |
| **Zone**          | `us-east4-c`                                         |
| **Image**         | `pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20251013` |

### 1. Connect to the VM
From your Cloud Shell or local terminal (with gcloud SDK installed)
```bash
gcloud config set project uml-gnn-transfer
gcloud compute ssh gnn-gpu-1 --zone=us-east4-c
```
The first time you connect, GCP will generate an SSH key automatically. Type Y when prompted.
### 2. Verify GPU Access
Once inside the VM:
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```
You should see the NVIDIA T4 listed and True printed by PyTorch.
### 3. Set Up Your Environment
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
git clone https://github.com/raavi02/UML-25.git
cd UML-25
pip install -r requirements.txt
```
### 4. Save Credits When Idle
Always stop the VM when you’re done:
```bash
gcloud compute instances stop gnn-gpu-1 --zone=us-east4-c
```
Restart later with:
```bash
gcloud compute instances start gnn-gpu-1 --zone=us-east4-c
```
