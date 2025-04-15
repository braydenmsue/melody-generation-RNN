import torch
import torch.nn as nn
from models.hyperparameters import HP


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx, dropout):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        # print("Input shape:", x.shape)
        # print("Input dtype:", x.dtype)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc(output)  # [batch_size, seq_len, vocab_size]
        return logits, hidden


