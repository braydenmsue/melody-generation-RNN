import torch
import torch.nn as nn


class HP:
    hidden_dim = 128
    embed_dim = 32
    n_layers = 1
    dropout = 0.2

    batch_size = 32
    num_epochs = 10
    lr = 0.001

    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    """
    GEEKSFORGEEKS EXAMPLE
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

    input_size = 1
    hidden_size = 20
    output_size = 1

    model = SimpleRNN(input_size, hidden_size, output_size)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out




