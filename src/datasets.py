# See kaggle dataset page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
import json
from os.path import join, exists
from pathlib import Path
from functools import partial

import rich
import rich.progress
import torch
import numpy as np
import pandas as pd
from kagglehub import dataset_download

from config import DATASET_HANDLE, SAMPLES_PER_NPY_FILE


# Make all track bars transient
_track = partial(rich.progress.track, transient=True)

class PreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, train=True, norm_input=True, norm_output=False, force_compute=False):
        super().__init__()
        self.train = train
        self.norm_input = norm_input
        self.norm_output = norm_output
        # column fold is equal to -100 for training and 0 for validation
        dataset_path = dataset_download(DATASET_HANDLE)
        fold_nb = str(-100 if train else 0)
        meta_df = (
            pd.read_csv(join(dataset_path, "folds.csv"))
            .query(f"fold == {fold_nb}")
            .reset_index(drop=True)
        )
        # load entirety of the dataset in the RAM
        self.x = [load_npy(dataset_path, f_path) for f_path in _track(meta_df["data_fpath"], "loading inputs")]
        self.y = [load_npy(dataset_path, f_path) for f_path in _track(meta_df["label_fpath"], "loading outputs")]

        self.set_stats(force_compute)

    def __getitem__(self, idx) -> torch.Tensor:
        list_idx = idx // SAMPLES_PER_NPY_FILE
        tensor_idx = idx % SAMPLES_PER_NPY_FILE
        return self.x[list_idx][tensor_idx], self.y[list_idx][tensor_idx]

    def __len__(self):
        return len(self.x) * SAMPLES_PER_NPY_FILE

    def set_stats(self, force_compute=False) -> dict[str, torch.Tensor]:
        stats_file_path = self.get_dataset_stats_file_path()
        if force_compute or not exists(stats_file_path):
            self.samples_stats = {}
            self.samples_stats["x_mean"], self.samples_stats["x_std"] = compute_welford_mean_std(self.x)
            self.samples_stats["y_mean"], self.samples_stats["y_std"] = compute_welford_mean_std(self.y)
            with open(stats_file_path, "w") as fp:
                json.dump(self.samples_stats, fp, indent=1)
        else:
            with open(stats_file_path, "r") as fp:
                self.samples_stats = json.load(fp)

    def get_dataset_stats_file_path(self) -> Path:
        suffix = "train" if self.train else "test"
        return (
            Path(__file__)
            .absolute()
            .parent
            .joinpath(f"dataset_stats_{suffix}.json")
        )

def compute_welford_mean_std(tensor_list):
    count = 0
    mean = torch.zeros(1, dtype=torch.float64)
    M2 = torch.zeros(1, dtype=torch.float64)

    for tensor in _track(tensor_list, description="Computing dataset stats"):
        x = tensor.view(-1).to(torch.float64)
        batch_count = x.numel()
        batch_mean = x.mean()
        batch_M2 = ((x - batch_mean) ** 2).sum()

        delta = batch_mean - mean
        total_count = count + batch_count

        mean += delta * batch_count / total_count
        M2 += batch_M2 + delta.pow(2) * count * batch_count / total_count
        count = total_count

    variance = M2 / count
    std = variance.sqrt().item()
    return mean.item(), std


def load_npy(dataset_path:str, path_in_dataset: str) -> torch.Tensor:
    path = join(dataset_path, 'openfwi_72x72', path_in_dataset)
    return torch.from_numpy(np.load(path))

def test_dataset(train:bool):
    dataset = PreprocessedOpenFWI(train, True)
    print("train:", train)
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
    print("=" * 20)

if __name__ == "__main__":
    test_dataset(True)
    test_dataset(False)