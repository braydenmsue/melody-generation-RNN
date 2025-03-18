import os
from models import rnn
import torch
from models.dataset import ABCDataset


def main(input_dir):

    json_path = os.path.join(input_dir, 'lookup_tables', 'songs_dict.json')
    # print(json_path)

    dataset = ABCDataset(json_path)
    # print(dataset.__len__())
    # print(dataset.__getitem__(0))

    training_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    return

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f'base dir: {base_dir}')
    data_path = os.path.join(base_dir, 'data')
    # print(f'data dir: {data_path}')

    main(data_path)

