import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dataset import ABCDataset
from models.train import train_model, eval_model, sample
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from models.hyperparameters import HP
import numpy as np
import random

def plot_scores(values, title="", ylabel="", xlabel="Sample"):
    plt.figure(figsize=(10, 4))
    x = range(len(values))

    plt.bar(x, values, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(x)

    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output/figs/{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch, pad_index):
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        inputs, targets, meta = zip(*batch)
    else:
        inputs, targets = zip(*batch)
        meta = None

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_index)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_index)

    return (inputs_padded, targets_padded, list(meta)) if meta else (inputs_padded, targets_padded)


def main(input_dir, train_flag=True, model_type='rnn'):
    json_path = os.path.join(input_dir, 'lookup_tables', 'songs_dict.json')
    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # print(json_path)
    
    set_seed(42)
    dataset = ABCDataset(json_path)
    model = None

    # return
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(train_ratio=0.8, val_ratio=0.1)

    # use predetermined pad_idx for padding in collate_fn function
    PAD_IDX = dataset.get_pad_idx()

    # dataloaders using the data resulting from split
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=HP.batch_size,
                                               shuffle=True,
                                               collate_fn=lambda b: collate_fn(b, PAD_IDX))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=HP.batch_size,
                                             shuffle=False,
                                             collate_fn=lambda b: collate_fn(b, PAD_IDX))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=HP.batch_size,
                                              shuffle=False,
                                              collate_fn=lambda b: collate_fn(b, PAD_IDX))

    # change to True to train model with hyperparameters from HP class
    NUM_SAMPLES = 10
    if train_flag:
        model = train_model(train_loader,
                            val_loader,
                            num_epochs=HP.num_epochs,
                            batch_size=HP.batch_size,
                            learning_rate=HP.lr,
                            mtype=model_type)
        filename = f"output/{model_type}_train.txt"
        samples = ""
        for i in range(NUM_SAMPLES):
            result, key_score, time_sig_score = sample(model, train_dataset)
            samples += f"SAMPLE {i + 1}:\n{result}\n\n"

        if filename:
            with open(filename, 'w') as f:
                f.write(samples)
            print(f"{NUM_SAMPLES} test samples written to {filename}.")

    if os.path.isfile(f"output/{model_type}_model.pth"):
        model = eval_model(test_loader, OUTPUT_DIR, model_type)
        filename = f"{OUTPUT_DIR}/test.txt"
        filename = "output/test.txt"
        samples = ""
        keyscores = []
        timescores = []

        for i in range(NUM_SAMPLES):
            s, k, t = sample(model, test_dataset)
            keyscores.append(k)
            timescores.append(t)
            samples += f"SAMPLE {i + 1}:\n{s}\n\n"

        avg_keyscore = sum(keyscores) / len(keyscores)
        avg_timescore = sum(timescores) / len(timescores)
        plot_scores(keyscores, title="Key-Adherence Scores", ylabel="In-Key (%)", xlabel="Sample")
        plot_scores(timescores, title="TS-Adherence Scores", ylabel="In-Time (%)", xlabel="Sample")
        print(f"Average Key-Adherence Score: {avg_keyscore}")
        print(f"Average TS-Adherence Score: {avg_timescore}")
        if filename:
            with open(filename, 'w') as f:
                f.write(samples)

            print(f"{NUM_SAMPLES} test samples written to {filename}.")
    else:
        print(f"'{OUTPUT_DIR}/{model_type}_model.pth' DOES NOT EXIST\n   Set `train_flag=True` in main.py.")

    # gen_output = sample(model, train_dataset)
    # print(f"\nOUTPUT SEQUENCE (TRAIN):\n{gen_output}")
    # gen_output = sample(model, test_dataset)
    # print(f"\nOUTPUT SEQUENCE (TEST):\n{gen_output}")
    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data')

    main(data_path, train_flag=True, model_type='gru')
