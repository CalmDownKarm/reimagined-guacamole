import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN1d(nn.Module):
    def __init__(self, embedding_dim=1, n_filters=3, filter_sizes=[2, 3, 4], output_dim=5,
                 dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embedding):
        """

        :param text_embedding: A batch_size *  d tensor where d is the length of sentence
        :return:
        """
        embedded = text_embedding.reshape(text_embedding.size(0), text_embedding.size(1), -1).cuda()

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

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