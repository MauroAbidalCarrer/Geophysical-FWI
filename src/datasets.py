# See kaggle competition page: https://www.kaggle.com/competitions/waveform-inversion
# See preprocessed test dataset: https://www.kaggle.com/datasets/egortrushin/open-wfi-test
# See kaggle preprocessed train and validation datasets page: https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72
from os.path import join

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from rich.progress import track
from kagglehub import dataset_download

from config import *


class TrainValidationPreprocessedOpenFWI(torch.utils.data.Dataset):
    def __init__(self, split:str="train", nb_files=None):
        super().__init__()
        self.x, self.y = _load_dataset_tensors(split, nb_files)

    def __getitem__(self, idx) -> torch.Tensor:
        list_idx = idx // SAMPLES_PER_NPY_FILE
        tensor_idx = idx % SAMPLES_PER_NPY_FILE
        return (
            self.x[list_idx][tensor_idx],
            self.y[list_idx][tensor_idx],
        )

    def __len__(self):
        return len(self.x) * SAMPLES_PER_NPY_FILE

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
    x = [_load_npy_fie(dataset_path, f_path) for f_path in tensors_it("data_fpath", "loading inputs")]
    y = [_load_npy_fie(dataset_path, f_path) for f_path in tensors_it("label_fpath", "loading outputs")]

    return x, y

def _load_npy_fie(dataset_path:str, path_in_dataset: str) -> torch.Tensor:
    path = join(dataset_path, 'openfwi_72x72', path_in_dataset)
    # return torch.from_numpy(np.load(path))
    return np.load(path, mmap_mode="r")

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
    print("=" * 20)

if __name__ == "__main__":  
    _check_dataset_shape("train")
    _check_dataset_shape("validation")
