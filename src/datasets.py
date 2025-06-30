# This module is a bit dirty, I am finishing it ~2 days before the end of the competition...
# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
# See preprocessed test dataset: https://www.kaggle.com/datasets/egortrushin/open-wfi-test
# See kaggle preprocessed train and validation datasets page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
import json
from os import listdir
from pathlib import Path
from typing import LiteralString
from os.path import join, exists, splitext

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from rich.progress import track
from kagglehub import dataset_download

from config import *


class TestPreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, force_stats_compute=False, nb_subset:int=None):
        super().__init__()
        self.stats = _get_train_stats("train", force_stats_compute)
        dataset_path = join(dataset_download(TEST_DATASET_HANDLE), "test")
        self.filenames = listdir(dataset_path)
        load_as_tensor = lambda filename: torch.from_numpy(np.load(join(dataset_path, filename)))
        nb_subset = nb_subset if nb_subset else len(self.filenames)
        files_it = track(self.filenames[:nb_subset], description="Loading test inputs...")
        self.x = list(map(load_as_tensor, files_it))

    def __getitem__(self, idx) -> tuple[str, torch.Tensor]:
        return (
            splitext(self.filenames[idx])[0],
            (self.x[idx] - self.stats["x_mean"]) / self.stats["x_std"],
        )

    def __len__(self):
        return len(self.filenames)

class TrainValidationPreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, split:str="train", nb_files=None, force_stats_compute=False):
        super().__init__()
        self.x, self.y = _load_dataset_tensors(split, nb_files)
        self.stats = _get_train_stats(split, force_stats_compute, self.x, self.y)
        null_std_mask = self.stats["x_std"] == 0
        self.stats["x_std"][null_std_mask] = 0.0001
        for i in range(len(self.x)):
            self.x[i] = (self.x[i] - self.stats["x_mean"]) / self.stats["x_std"]
            self.y[i] = (self.y[i] - self.stats["y_mean"]) / self.stats["y_std"]

    def __getitem__(self, idx) -> torch.Tensor:
        list_idx = idx // SAMPLES_PER_NPY_FILE
        tensor_idx = idx % SAMPLES_PER_NPY_FILE
        return (
            self.x[list_idx][tensor_idx],
            self.y[list_idx][tensor_idx],
        )

    def __len__(self):
        return len(self.x) * SAMPLES_PER_NPY_FILE

def _get_train_stats(split:str, force_compute=False, x:list[Tensor]=None, y:list[Tensor]=None) -> dict:
    stats_file_path = _get_dataset_stats_file_path(split)
    if force_compute or not exists(stats_file_path):
        stats = {}
        if x is None or y is None:
            x, y = _load_dataset_tensors("train")
        stats["x_mean"], stats["x_std"] = compute_welford_mean_std(x)
        stats["y_mean"], stats["y_std"] = compute_welford_mean_std(y)
        save_stats_to_json(stats, stats_file_path)
    else:
        return load_stats_from_json(stats_file_path)


def save_stats_to_json(stats: dict[str, Tensor], filepath:str):
    # Convert tensors to nested lists for JSON compatibility
    serializable_stats = {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in stats.items()
    }
    with open(filepath, "w") as fp:
        json.dump(serializable_stats, fp, indent=1)

def load_stats_from_json(filepath) -> dict[str, Tensor]:
    with open(filepath, "r") as fp:
        raw_stats = json.load(fp)
    # Convert lists back to tensors
    stats = {
        key: torch.tensor(value)
        for key, value in raw_stats.items()
    }
    return stats

def _load_dataset_tensors(split:str, nb_files=None) -> tuple[Tensor, Tensor]:
    dataset_path = dataset_download(TRAIN_VALIDAION_DATASET_HANDLE)
    # column "fold" is equal to -100 for training and 0 for validation see kaggle dataset description
    if split == "train":
        fold_nb = -100 
    elif split == "validation":
        fold_nb = 0
    else:
        raise NotImplementedError(f'{split} is not a valid split, either use "train" or "validation"')
    meta_df = (
        pd.read_csv(join(dataset_path, "folds.csv"))
        .query(f"fold == {fold_nb}")
        .reset_index(drop=True)
    )
    nb_files = nb_files if nb_files else len(meta_df)
    tensors_it = lambda path, files_group: track(meta_df.loc[:nb_files, path], files_group)
    x = [_load_npy_file_as_tensor(dataset_path, f_path) for f_path in tensors_it("data_fpath", "loading inputs")]
    y = [_load_npy_file_as_tensor(dataset_path, f_path) for f_path in tensors_it("label_fpath", "loading outputs")]

    return x, y

def _load_npy_file_as_tensor(dataset_path:str, path_in_dataset: str) -> torch.Tensor:
    path = join(dataset_path, 'openfwi_72x72', path_in_dataset)
    return torch.from_numpy(np.load(path))

def _get_dataset_stats_file_path(split:str) -> Path:
    return (
        Path(__file__)
        .absolute()
        .parent
        .joinpath(f"dataset_stats_{split}.json")
    )

def compute_welford_mean_std(tensor_list: list[Tensor]) -> dict[str, Tensor]:
    first = next(iter(tensor_list))
    mean = torch.zeros_like(first[0], dtype=torch.float64)
    M2 = torch.zeros_like(first[0], dtype=torch.float64)
    count = 0

    # Re-inject the first batch
    tensor_list = [first] + list(tensor_list)

    for batch in track(tensor_list, description="Computing pixel-wise stats"):
        batch = batch.to(torch.float64)  # shape: (N, C, H, W)
        batch_count = batch.shape[0]

        batch_mean = batch.mean(dim=0)  # shape: (C, H, W)
        batch_M2 = ((batch - batch_mean)**2).sum(dim=0)  # shape: (C, H, W)

        delta = batch_mean - mean
        total_count = count + batch_count

        mean += delta * batch_count / total_count
        M2 += batch_M2 + delta.pow(2) * count * batch_count / total_count

        count = total_count

    variance = M2 / count
    std = variance.sqrt()

    return mean.float(), std.float()

def _check_dataset_shape(split:str):
    dataset = TrainValidationPreprocessedOpenFWI(split)
    print("split:", split)
    print(dataset)
    print("len:", len(dataset))
    def test_sample(idx:int):
        x, y = dataset[0]
        print(f"x {idx} data type:", type(x))
        print(f"x {idx} shape:", x.shape)
        print(f"y {idx} data type:", type(y))
        print(f"y {idx} shape:", y.shape)

    test_sample(0)
    test_sample(100)
    test_sample(len(dataset) - 1)
    print("stats:", dataset.stats)
    print("=" * 20)

if __name__ == "__main__":
    test_dataset = TestPreprocessedOpenFWI()
    print("test stats(train stats):", test_dataset.stats)
    first_test_filename, first_test_input = test_dataset[0]
    print("First test sample filename:", first_test_filename)
    print("First test sample filename:", first_test_input.shape)
    
    _check_dataset_shape("train")
    _check_dataset_shape("validation")
    