import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import ABCDataset
from torch.utils.data import Subset
from models.rnn import RNNModel, HP
import os
import wandb


OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
# print(OUTPUT_DIR)


def train_model(dataloader, num_epochs=3, batch_size=32, learning_rate=0.0005):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = HP.device

    # input_dir = dataloader.dataset.file_name
    # json_path = f"{input_dir}/lookup_tables/songs_dict.json"
    # dataset = ABCDataset(json_path)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    # input_size = dataloader.dataset.sequences[0].shape[2]  # VOCAB_SIZE
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    input_size = dataset.get_vocab_size()
    PAD_IDX = dataset.get_pad_idx()

    hidden_size = HP.hidden_dim
    output_size = input_size
    model = RNNModel(input_size, hidden_size, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # run = wandb.init(
    # # entity="mga113-org",
    # # project="melody_generation"
    # # name="CMPT 413 - Melody Generation",
    # )   
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            model.zero_grad()

            batch_size, seq_length, _ = input_tensor.shape
            hidden = torch.zeros(1, batch_size, hidden_size).to(device)

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

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/rnn_model.pth")
    # torch.save(model.state_dict(), f"{OUTPUT_DIR}/rnn_model_{HP.num_epochs}.pth")
    print("Training complete. Model saved.")

    return model


def eval_model(test_loader):
    dataset = test_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    input_size = dataset.get_vocab_size()
    PAD_IDX = dataset.get_pad_idx()

    hidden_size = HP.hidden_dim
    output_size = input_size
    model = RNNModel(input_size, hidden_size, output_size).to(HP.device)

    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/rnn_model.pth"))
    device = HP.device
    model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

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

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")