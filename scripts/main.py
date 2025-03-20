import os
from models.rnn import HP
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dataset import ABCDataset, char2ind
from models.train import train_model, eval_model
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, pad_index):
    inputs, targets = zip(*batch)

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_index)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_index)

    inputs_padded = inputs_padded.squeeze(2)
    targets_padded = targets_padded.squeeze(2)
    # print(f"Inputs padded shape: {inputs_padded.shape}")
    # print(f"Targets padded shape: {targets_padded.shape}")
    return inputs_padded, targets_padded


def main(input_dir, train_flag=False):

    json_path = os.path.join(input_dir, 'lookup_tables', 'songs_dict.json')
    # print(json_path)

    dataset = ABCDataset(json_path)
    model = None

    test_ratio = 0.2  # 80/20 train/test split
    train_dataset, test_dataset = dataset.split_dataset(test_ratio)
    # print(dataset.__len__())
    # print(dataset.__getitem__(0))

    PAD_IDX = dataset.get_pad_idx()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=HP.batch_size,
                                               shuffle=True,
                                               collate_fn=lambda b: collate_fn(b, PAD_IDX))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=HP.batch_size,
                                              shuffle=True,
                                              collate_fn=lambda b: collate_fn(b, PAD_IDX))

    if train_flag:
        model = train_model(train_loader, num_epochs=HP.num_epochs, batch_size=HP.batch_size, learning_rate=HP.lr)
    eval_model(test_loader)

    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f'base dir: {base_dir}')
    data_path = os.path.join(base_dir, 'data')
    # print(f'data dir: {data_path}')

    main(data_path, True)

