import os
from models.rnn import HP
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dataset import ABCDataset
from models.train import train_model, eval_model, sample
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
    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # print(json_path)

    dataset = ABCDataset(json_path)
    model = None

    test_ratio = 0.2  # fraction of data used for test set
    train_dataset, test_dataset = dataset.split_dataset(test_ratio)

    # use predetermined pad_idx for padding in collate_fn function
    PAD_IDX = dataset.get_pad_idx()

    # dataloaders using the data resulting from split
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=HP.batch_size,
                                               shuffle=True,
                                               collate_fn=lambda b: collate_fn(b, PAD_IDX))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=HP.batch_size,
                                              shuffle=True,
                                              collate_fn=lambda b: collate_fn(b, PAD_IDX))

    # change to True to train model with hyperparameters from HP class
    NUM_SAMPLES = 10
    if train_flag:
        model = train_model(train_loader,
                            OUTPUT_DIR,
                            num_epochs=HP.num_epochs,
                            batch_size=HP.batch_size,
                            learning_rate=HP.lr)
        filename = f"{OUTPUT_DIR}/train.txt"
        samples = ""
        for i in range(NUM_SAMPLES):
            samples += f"SAMPLE {i+1}:\n{sample(model, train_dataset)}\n\n"
        if filename:
            with open(filename, 'w') as f:
                f.write(samples)
            print(f"{NUM_SAMPLES} test samples written to {filename}.")

    if os.path.isfile(f"{OUTPUT_DIR}/rnn_model.pth"):
        model = eval_model(test_loader, OUTPUT_DIR)
        filename = f"{OUTPUT_DIR}/test.txt"
        samples = ""
        for i in range(NUM_SAMPLES):
            samples += f"SAMPLE {i+1}:\n{sample(model, test_dataset)}\n\n"
        if filename:
            with open(filename, 'w') as f:
                f.write(samples)
            print(f"{NUM_SAMPLES} test samples written to {filename}.")
    else:
        print(f"'{OUTPUT_DIR}/rnn_model.pth' DOES NOT EXIST\n   Set `train_flag=True` in main.py.")

    # gen_output = sample(model, train_dataset)
    # print(f"\nOUTPUT SEQUENCE (TRAIN):\n{gen_output}")
    # gen_output = sample(model, test_dataset)
    # print(f"\nOUTPUT SEQUENCE (TEST):\n{gen_output}")
    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data')

    main(data_path, train_flag=True)

