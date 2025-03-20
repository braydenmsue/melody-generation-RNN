import torch
import torch.nn as nn


class HP:
    hidden_dim = 128
    embed_dim = 32
    n_layers = 1
    dropout = 0.2

    batch_size = 32
    num_epochs = 30
    lr = 0.005
    output_len = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        # TODO: figure out LSTM
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), HP.hidden_dim).to(x.device)
        
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden





