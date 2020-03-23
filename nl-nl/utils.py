import pandas as pd
import os
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch
import numpy as np

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


def read_imdb(data_dir, colnames=[TEXT_COL, LABEL_COL]):
    datasets = {}
    df = pd.read_csv(data_dir + "/IMDB Dataset.csv", header=0, names=colnames)
    df.loc[df[LABEL_COL] == "positive", LABEL_COL] = 1
    df.loc[df[LABEL_COL] == "negative", LABEL_COL] = 0
    train_split, dev_split, test_split = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

    for (t, split) in [("train", train_split), ("dev", dev_split), ("test", test_split)]:
        dataframe = split
        dataframe[LABEL_COL] = dataframe[LABEL_COL].astype(int)   # Categorical data type for truth labels
        datasets[t] = dataframe

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