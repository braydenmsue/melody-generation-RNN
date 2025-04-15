import torch


class HP:
    hidden_dim = 144 
    embed_dim = 144
    n_layers = 4    # set to 2 for LSTM
    # n_layers = 2
    dropout = 0.5

    batch_size = 32
    num_epochs = 200    # with early stopping, i.e. upper bound
    lr = 0.0006 
    output_len = 256
    
    # > 1 for more creative < 1 for more rule-following
    temp = 1.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
