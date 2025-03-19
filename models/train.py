import torch
import torch.nn as nn
import torch.optim as optim
from models.dataset import ABCDataset
from models.rnn import RNNModel
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, targets = zip(*batch)  # Only two values (not three)

    # Pad sequences so they have the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    inputs_padded= inputs_padded.squeeze(2)
    targets_padded = targets_padded.squeeze(2)

    # print(f"inputs_padded shape: {inputs_padded.shape}")  # Debugging
    # print(f"targets_padded shape: {targets_padded.shape}")  # Debugging

    return inputs_padded, targets_padded


def train_model(input_dir, num_epochs=5, batch_size=4, learning_rate=0.0005):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    json_path = f"{input_dir}/lookup_tables/songs_dict.json"
    dataset = ABCDataset(json_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    # Initialize model
    input_size = dataset.sequences[0].shape[2]  # VOCAB_SIZE
    hidden_size = 128
    output_size = input_size  # Predicts the next character in sequence
    model = RNNModel(input_size, hidden_size, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            model.zero_grad()

            # Reshape for batch processing
            batch_size, seq_length, _ = input_tensor.shape
            hidden = torch.zeros(1, batch_size, hidden_size).to(device)

            loss = 0
            for i in range(seq_length):
                output, hidden = model(input_tensor[:, i, :].unsqueeze(1), hidden)
                output = output.squeeze(1)  # Remove extra dimension
                loss += criterion(output, target_tensor[:, i, :])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), f"{input_dir}/rnn_model.pth")
    print("Training complete. Model saved.")

    return model


