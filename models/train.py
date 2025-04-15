import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.distributions import Categorical
from models.rnn import RNNModel
from models.hyperparameters import HP
from models.LSTMModel import LSTMModel
from models.dataset import tok2ind
from models.GRU_model import GRUModel
from models.tests import evaluate_key, evaluate_time_signature
import copy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_type = 'lstm'


def sample(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    idx2char_dict = dataset.idx2char_dict

    rand_idx = np.random.randint(len(dataset) - 1)
    input_seq, _, input_metadata = dataset[rand_idx]

    T = input_metadata.get('T:', "")
    L = input_metadata.get('L:', "")
    K = input_metadata.get('K:', "")

    SPECIAL = {'[PAD]', '[UNK]', '[START]', '[END]'}

    metadata_tokens = [f"T:{T}", f"L:{L}", f"K:{K}"]
    metadata_indices = [tok2ind(tok) for tok in metadata_tokens]
    input_seq = input_seq.unsqueeze(0).to(device)
    metadata_tensor = torch.tensor(metadata_indices, dtype=torch.long).unsqueeze(0).to(device)
    input_seq = torch.cat([metadata_tensor, input_seq], dim=1)

    hidden_state = None

    generated = metadata_tokens.copy()
    with torch.no_grad():
        output, hidden_state = model(input_seq, hidden_state)

        last_output = output[:, -1, :]
        vals = last_output / HP.temp
        probs = F.softmax(vals, dim=1)

        dist = Categorical(probs)
        next_token = dist.sample()

        current_input = next_token.unsqueeze(0)

        for _ in range(HP.output_len):
            output, hidden_state = model(current_input, hidden_state)
            last_output = output[:, -1, :]

            vals = last_output / HP.temp
            probs = F.softmax(vals, dim=1)

            dist = Categorical(probs)
            next_token = dist.sample()

            token_idx = next_token.item()
            if token_idx in idx2char_dict:
                tok = idx2char_dict[token_idx]
                if tok not in SPECIAL:
                    generated.append(tok)

            current_input = next_token.unsqueeze(0)

    result = ' '.join(generated)

    if K and L and T:
        key_score = evaluate_key(result, K)
        time_sig_score = evaluate_time_signature(result, L, T)
        # print(f"Key: {K}, Score: {key_score:.2%}")
        # print(f"Time Signature: {T}, Length: {L}, Score: {time_sig_score}")
        return result, key_score, time_sig_score

    elif K:
        key_score = evaluate_key(result, K)
        # print(f"Key: {K}, Score: {key_score:.2f%}")
        return result, key_score, 0.0

    elif L and T:
        time_sig_score = evaluate_time_signature(result, L, T)
        # print(f"Time Signature: {T:.2f%}, Length: {L:.2f%}, Score: {time_sig_score:.2f%}")
        return result, 0.0, time_sig_score

    return result, 0.0, 0.0


def train_model(dataloader, val_loader, num_epochs=3, batch_size=32, learning_rate=0.0005, mtype='rnn'):
    device = HP.device
    dataset = dataloader.dataset

    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    input_size = dataset.get_vocab_size()
    hidden_size = HP.hidden_dim
    output_size = input_size

    model = None
    if mtype == 'rnn':
        model = RNNModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)
    elif mtype == 'lstm':
        model = LSTMModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)
    elif mtype == 'gru':
        model = GRUModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)

    PAD_IDX = dataset.get_pad_idx()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    best_model = None
    epochs_no_improvement = 0
    records = []
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for input_tensor, target_tensor, _ in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            optimizer.zero_grad()

            output, _ = model(input_tensor)

            output = output.reshape(-1, output_size)
            target_flat = target_tensor.reshape(-1)

            loss = criterion(output, target_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_tensor, target_tensor, _ in val_loader:
                input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

                output, _ = model(input_tensor)

                output = output.reshape(-1, output_size)
                target_flat = target_tensor.reshape(-1)

                loss = criterion(output, target_flat)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        records.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

        # early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improvement = 0
            best_model = copy.deepcopy(model.state_dict())
            print(f"Model saved @ epoch {epoch + 1}")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= 5:
                print("Early stopping triggered.")
                break

    if best_model is not None:
        torch.save(model.state_dict(), f"output/{mtype}_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f"{mtype.upper()} Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/figs/{mtype}_loss.png")
    plt.close()

    records_df = pd.DataFrame(records)
    records_df.to_csv(f"output/{mtype}_loss.csv", index=False)
    print("Training complete. Model saved.")

    return model


def eval_model(test_loader, output_dir, model_type='rnn'):
    dataset = test_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    input_size = dataset.get_vocab_size()
    hidden_size = HP.hidden_dim
    output_size = input_size
    device = HP.device

    model = None
    if model_type == 'rnn':
        model = RNNModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)
    if model_type == 'lstm':
        model = LSTMModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)
    if model_type == 'gru':
        model = GRUModel(input_size, HP.embed_dim, hidden_size, HP.n_layers, dataset.get_pad_idx(), HP.dropout).to(device)

    model.load_state_dict(torch.load(f"output/{model_type}_model.pth"))
    model.to(device)
    model.eval()

    PAD_IDX = dataset.get_pad_idx()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    test_loss = 0
    correct = 0
    total = 0

    losses = []
    accuracies = []
    with torch.no_grad():
        for input_tensor, target_tensor, metadata in test_loader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            output, _ = model(input_tensor)

            output_flat = output.reshape(-1, output_size)
            target_flat = target_tensor.reshape(-1)

            loss = criterion(output_flat, target_flat)
            test_loss += loss.item()
            losses.append(loss.item())

            preds = output_flat.argmax(dim=1)
            mask = (target_flat != PAD_IDX)
            correct_batch = (preds[mask] == target_flat[mask]).sum().item()
            total_batch = mask.sum().item()
            acc = 100.0 * correct_batch / total_batch
            accuracies.append(acc)
            total += total_batch
            correct += correct_batch

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")

    plt.figure(figsize=(10, 5))
    x = range(len(losses))
    plt.bar(x, losses, edgecolor='black')
    plt.title('Test Losses')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f"output/figs/test_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    x = range(len(accuracies))
    plt.bar(x, accuracies, edgecolor='black')
    plt.title('Test Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f"output/figs/test_accuracy.png")
    plt.close()

    return model
