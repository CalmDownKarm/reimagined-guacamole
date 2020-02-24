import pandas as pd
import os
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch

TEXT_COL, LABEL_COL = 'text', 'sentiment'


def read_sst5(data_dir, colnames=[LABEL_COL, TEXT_COL]):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"sst_{t}.txt"), sep='\t', header=None, names=colnames)
        df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
        df[LABEL_COL] = df[LABEL_COL].astype(int)   # Categorical data type for truth labels
        df[LABEL_COL] = df[LABEL_COL] - 1  # Zero-index labels for PyTorch
        datasets[t] = df
    return datasets


def create_dataloader(features, labels, batch_size=32, shuffle=False):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.long), #float
                            torch.tensor(labels, dtype=torch.long))

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=shuffle,
                             pin_memory=torch.cuda.is_available())

    return data_loader