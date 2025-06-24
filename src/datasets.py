# See kaggle dataset page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
import json
from math import sqrt
from os.path import join, exists

import torch
import numpy as np
import pandas as pd
from rich.progress import track
from kagglehub import dataset_download

from config import DATASET_HANDLE, SAMPLES_PER_NPY_FILE


DATASET_PATH = dataset_download(DATASET_HANDLE)

class PreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, train=True, force_samples_stats_compute=False):
        self.train = train
        # column fold is equal to -100 for training and 0 for validation
        fold_nb = str(-100 if train else 0)
        meta_df = (
            pd.read_csv(join(DATASET_PATH, "folds.csv"))
            .query(f"fold == {fold_nb}")
            .reset_index(drop=True)
        )
        # load entirety of the dataset in the RAM
        self.x = [load_npy(path) for path in track(meta_df["data_fpath"], "loading xs")]
        print("concatenating xs")
        self.x = torch.concatenate(self.x)
        print("x:", self.x.shape)
        self.y = [load_npy(path) for path in track(meta_df["label_fpath"], "loading ys")]
        print("concatenating ys")
        self.y = torch.concatenate(self.y)
        print("y:", self.y.shape)

        self.samples_stats = {
            "x_mean": self.x.mean(),
            "x_std": self.x.std(),
            "y_mean": self.y.mean(),
            "y_std": self.y.std(),
        }
        print(self.samples_stats)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x) * SAMPLES_PER_NPY_FILE

def load_npy(path_in_dataset: str) -> np.ndarray:
    path = join(DATASET_PATH, 'openfwi_72x72', path_in_dataset)
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