import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import ABCDataset, line2tensor
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.distributions import Categorical
from models.rnn import RNNModel, HP
import os
import numpy as np
import wandb


def sample(model, dataset):

    model.eval()

    data_ptr = 0
    hidden_state = None

    rand_idx = np.random.randint(len(dataset) - 1)
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    data = dataset.sequences

    # select a random sequence
    input_seq = data[rand_idx:rand_idx + 1]     # (sequence_length, batch_size, input_size)
    input_seq = input_seq[0].permute(1, 0, 2)   # (batch_size, sequence_length, input_size)

    generated = []
    while True:

        output, hidden_state = model(input_seq, hidden_state)
        # print(f"probs shape before squeeze: {output.shape}")

        output = output[:, -1, :]  # (batch_size, input_size)
        output = F.softmax(output, dim=-1)

        dist = Categorical(output)
        index = dist.sample()

        # print(dataset.idx2char_dict[index.item()], end='')
        generated.append(dataset.idx2char_dict[index.item()])
        input_seq[0][0] = index.item()
        data_ptr += 1
        if data_ptr > HP.output_len:
            break

    result = ''.join(generated)

    return result


def train_model(dataloader, output_dir, num_epochs=3, batch_size=32, learning_rate=0.0005):

    device = HP.device
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    input_size = dataset.get_vocab_size()

    hidden_size = HP.hidden_dim
    output_size = input_size
    model = RNNModel(input_size, hidden_size, output_size).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # run = wandb.init(
    # # entity="mga113-org",
    # # project="melody_generation"
    # # name="CMPT 413 - Melody Generation",
    # )

    PAD_IDX = dataset.get_pad_idx()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            model.zero_grad()

            batch_size, seq_length, _ = input_tensor.shape

            output, hidden = model(input_tensor)

            output = output.reshape(-1, output_size)

            target_indices = torch.argmax(target_tensor, dim=2).reshape(-1)
            loss = criterion(output, target_indices)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    #     wandb.log({
    #                 "train_loss": avg_loss
    #             })
    # run.finish()

    torch.save(model.state_dict(), f"{output_dir}/rnn_model.pth")
    # torch.save(model.state_dict(), f"{OUTPUT_DIR}/rnn_model_{HP.num_epochs}.pth")
    print("Training complete. Model saved.")

    return model


def eval_model(test_loader, output_dir):
    dataset = test_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    input_size = dataset.get_vocab_size()

    hidden_size = HP.hidden_dim
    output_size = input_size
    model = RNNModel(input_size, hidden_size, output_size).to(HP.device)

    model.load_state_dict(torch.load(f"{output_dir}/rnn_model.pth"))
    device = HP.device
    model.to(device)
    model.eval()

    PAD_IDX = dataset.get_pad_idx()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_tensor, target_tensor in test_loader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output, _ = model(input_tensor)
            output = output.reshape(-1, output_size)

            target_indices = torch.argmax(target_tensor, dim=2).reshape(-1)

            loss = criterion(output, target_indices)
            test_loss += loss.item()

            pred = output.argmax(dim=1)

            mask = (target_indices != PAD_IDX)
            correct += (pred[mask] == target_indices[mask]).sum().item()
            total += mask.sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return model
