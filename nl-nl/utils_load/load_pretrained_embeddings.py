import io
import numpy as np


def load_vectors(location):
    fin = io.open(location, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    vocab = []
    embedding = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embedding[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
        vocab.append(tokens[0])
    return embedding, vocab


def load_word_idx_maps(vocab):
    word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
    index = 4
    for word in vocab:
        word2idx[word] = index
        idx2word[index] = word
        index = index + 1
    return word2idx, idx2word


def load_lookup_table(embedding, word2idx):
    lookup_table = {}
    rows = cols = 0
    pretrained_vectors = []
    for (word, vector) in embedding.items():
        index = word2idx[word]
        lookup_table[index] = vector
        pretrained_vectors.append(vector)
        (cols,) = vector.shape
    initial_vectors = np.random.randn(4, cols)
    pretrained_vectors = np.vstack((pretrained_vectors, initial_vectors))
    return lookup_table, pretrained_vectors
