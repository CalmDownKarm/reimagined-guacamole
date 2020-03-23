import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN1d(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=300, n_filters=3, filter_sizes=[2, 4, 6, 8], output_dim=5,
                     dropout=0.1, pad_idx=None, pre_trained_embedding = None):
        super().__init__()

        self.weights = torch.FloatTensor(pre_trained_embedding)

        self.embedding = nn.Embedding.from_pretrained(self.weights)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
            ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """

        :param text_embedding: A batch_size *  d tensor where d is the length of sentence
        :return:
        """
        # text = [batch size, sent len]

        embedded = self.embedding(text.cuda())

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]
        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        linear_output = self.fc(cat)

        probabilties = F.log_softmax(linear_output)

        return probabilties