# root/model/gnn_refiner.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging
import argparse
import yaml
import os
import copy

# Data loading utilities
from src.utils.io import project_root, load_cfgs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Main GNN Pipeline
# --------------------------------------------------------------------------- #

def pipeline(config_file: str, for_seed: int | None = None) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------

    # load cfg (pipeline yaml) and cfg_lp (lp yaml)
    cfg_pipe, cfg_d = load_cfgs(config_file)  # cfg_lp is a DictConfig, cfg_pipe is not

    # Define directories
    dataset = cfg_pipe.dataset_name
    data_dir = cfg_pipe.dataset_dir
    outputs_dir = cfg_pipe.outputs_dir
    gt_dir = os.path.join(data_dir, f'{dataset}_ground_truth')
    preds_dir = os.path.join(data_dir, f'{dataset}_predictions')

    # Dataset parameters
    cameras = cfg_d.data.view_names
    keypoints = cfg_d.data.keypoint_names
    skeleton = cfg_d.data.skeleton

    print(f'Dataset: {dataset}')
    print(f'Data Directory: {data_dir}')
    print(f'Output Directory: {outputs_dir}')
    print(f'Ground Truth Directory: {gt_dir}')
    print(f'Predictions Directory: {preds_dir}')
    print(f'Camera views: {cameras}')
    print(f'Keypoints: {keypoints}')
    print(f'Bones: {skeleton}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    args = parser.parse_args()
    pipeline(args.config)
