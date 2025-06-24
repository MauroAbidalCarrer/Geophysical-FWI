# See kaggle dataset page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
from os.path import join

import torch
import numpy as np
import pandas as pd
import plotly.express as px
from kagglehub import dataset_download

from config import DATASET_HANDLE, SAMPLES_PER_NPY_FILE


DATASET_PATH = dataset_download(DATASET_HANDLE)
print(DATASET_PATH)

class PreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.train = train
        # column fold is equal to -100 for training and 0 for validation
        fold_nb = str(-100 if train else 0)
        self.metad_data = (
            pd.read_csv(join(DATASET_PATH, "folds.csv"))
            .query(f"fold == {fold_nb}")
            .reset_index(drop=True)
        )
        print(self.metad_data)

    def __getitem__(self, idx):
        meta_data_row_idx = idx // SAMPLES_PER_NPY_FILE
        sample_idx_in_array = idx % SAMPLES_PER_NPY_FILE
        x_path, y_path, family, _ = self.metad_data.iloc[meta_data_row_idx]
        return (
            np.load(join(DATASET_PATH, 'openfwi_72x72', x_path))[sample_idx_in_array],
            np.load(join(DATASET_PATH, 'openfwi_72x72', y_path))[sample_idx_in_array],
        )

    def __len__(self, ):
        return len(self.metad_data) * SAMPLES_PER_NPY_FILE


def test_dataset(train:bool):
    dataset = PreprocessedOpenFWI(train)
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