"""
Taken from the blog :- https://towardsdatascience.com/fine-grained-sentiment-analysis-part-3-fine-tuning-transformers-1ae6574f25a6
Adapted from Oliver Atanaszov's notebook on transformer fine-tuning
https://github.com/ben0it8/containerized-transformer-finetuning/blob/develop/research/finetune-transformer-on-imdb5k.ipynb
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm
from typing import Tuple, Any
import string

n_cpu = multiprocessing.cpu_count()
MAX_LENGTH = 256

TEXT_COL, LABEL_COL = 'text', 'sentiment'


class TextProcessor:
    def __init__(self, tokenizer, label2id, word2idx, idx2word, vocab, max_length=512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab = vocab
        self.start_token = "<START>"  # indicates end of sentence
        self.pad_token = "<PAD>"  # indicates padding
        self.end_token = "<END>"  # indicates end token
        self.unk_token = "<UNK>"  # unknown token

    def encode(self, input):
        return list(self.word2idx[o] for o in input)

    def token2id(self, item: Tuple[str, str]):
        """Convert text (item[0]) to sequence of IDs and label (item[1]) to integer"""
        assert len(item) == 2  # Need a row of text AND labels
        label, text = item[0], item[1]
        assert isinstance(text, str)  # Need position 1 of input to be of type(str)
        inputs = [self.start_token]

        for token in self.tokenizer(text.lower().translate(str.maketrans('', '', string.punctuation)).strip()):
            word = token
            if word in self.vocab:
                inputs.append(word)
            else:
                inputs.append(self.unk_token)

        # Trim or pad dataset
        if len(inputs) >= self.max_length:
            inputs = inputs[:self.max_length - 1]
            ids = self.encode(inputs) + [self.word2idx[self.end_token]]
        else:
            pad = [self.word2idx[self.pad_token]] * (self.max_length - len(inputs) - 1)
            ids = self.encode(inputs) + [self.word2idx[self.end_token]] + pad

        return np.array(ids, dtype='int64'), self.label2id[label]

    def process_row(self, row):
        """Calls the token2id method of the text processor for passing items to executor"""
        return self.token2id((row[1][LABEL_COL], row[1][TEXT_COL]))

    def create_dataloader(self,
                          df: pd.DataFrame,
                          batch_size: int = 32,
                          shuffle: bool = False,
                          valid_pct: float = None):
        """Process rows in pd.DataFrame using n_cpus and return a DataLoader"""

        tqdm.pandas()
        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            result = list(
                tqdm(executor.map(self.process_row, df.iterrows(), chunksize=8192),
                     desc=f"Processing {len(df)} examples on {n_cpu} cores",
                     total=len(df)))

        features = [r[0] for r in result]
        labels = [r[1] for r in result]

        dataset = TensorDataset(torch.tensor(features, dtype=torch.long),  # float
                                torch.tensor(labels, dtype=torch.long))

        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=shuffle,
                                 pin_memory=torch.cuda.is_available())
        return data_loader