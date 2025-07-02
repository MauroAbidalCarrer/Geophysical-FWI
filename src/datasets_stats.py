"""
Module to load the dataset stats.
Look at the [stats computation notebook](notebooks/compute_train_datasets_stats.ipynb) to see how they are computed.
"""
import json

import torch
from torch import Tensor
from kagglehub import dataset_download

from config import DATASET_STATS_HANDLE


stats_type = dict[str, dict[str, dict[str, Tensor]]]

def get_training_stats() -> stats_type:
    stats_path = dataset_download(DATASET_STATS_HANDLE)
    return load_stats_from_json(stats_path)

def load_stats_from_json(filepath: str) -> stats_type:
    with open(filepath, "r") as fp:
        serialized_stats = json.load(fp)
    # Convert lists back to tensors
    return {
        "pixel_wise_stats": {
            "x": {key: torch.tensor(value) for key, value in serialized_stats["pixel_wise_stats"]["x"].items()},
            "y": {key: torch.tensor(value) for key, value in serialized_stats["pixel_wise_stats"]["y"].items()},
        },
        "pixel_agnostic_stats": serialized_stats["pixel_agnostic_stats"],
    }
