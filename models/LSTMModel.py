import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_idx, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        # print("Input shape:", x.shape)
        # print("Input dtype:", x.dtype)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)  # [batch_size, seq_len, vocab_size]
        return logits, hidden

