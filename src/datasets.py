# See kaggle dataset page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
# from statistics import mean
# from math import mean
from os.path import join
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
    def __init__(self, train=True):
        self.train = train
        # column fold is equal to -100 for training and 0 for validation
        dataset_path = dataset_download(DATASET_HANDLE)
        fold_nb = str(-100 if train else 0)
        meta_df = (
            pd.read_csv(join(dataset_path, "folds.csv"))
            .query(f"fold == {fold_nb}")
            .reset_index(drop=True)
        )
        # load entirety of the dataset in the RAM
        self.x = [load_npy(dataset_path, f_path) for f_path in _track(meta_df["data_fpath"], "loading xs")]
        self.y = [load_npy(dataset_path, f_path) for f_path in _track(meta_df["label_fpath"], "loading ys")]

        self.samples_stats = {}
        self.samples_stats["x_mean"], self.samples_stats["x_std"] = welford_mean_std(self.x)
        self.samples_stats["y_mean"], self.samples_stats["y_std"] = welford_mean_std(self.y)
        print(self.samples_stats)

    def __getitem__(self, idx) -> torch.Tensor:
        list_idx = idx // SAMPLES_PER_NPY_FILE
        tensor_idx = idx % SAMPLES_PER_NPY_FILE
        return self.x[list_idx][tensor_idx], self.y[list_idx][tensor_idx]

    def __len__(self):
        return len(self.x) * SAMPLES_PER_NPY_FILE

def welford_mean_std(tensor_list):
    count = 0
    mean = torch.zeros(1, dtype=torch.float64)
    M2 = torch.zeros(1, dtype=torch.float64)

    for tensor in _track(tensor_list, description="Computing stats"):
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