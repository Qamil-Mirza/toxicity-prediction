from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch
from models.MultiTaskToxicityModel import masked_bce_loss
import yaml

# Load the config file
with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def train_model(model, optimizer, X_train, y_train, batch_size):
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    # Create a binary mask (1 for valid labels, 0 for NaNs in y_train)
    mask_tensor = torch.isnan(y_train_tensor).logical_not().float()

    # Replace NaNs in y_train_tensor with zeros (they won't contribute to loss due to masking)
    y_train_tensor[torch.isnan(y_train_tensor)] = 0

    # Create a TensorDataset and DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor, mask_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Set model to training mode
    model.train()

    epoch_losses = []  # Track loss for each epoch
    epochs = config["hyperparameters"]["epochs"]
    # Begin training loop
    for epoch in range(epochs):
        total_loss = 0  # Accumulate loss over all batches in this epoch

        for i, (X_batch, y_batch, mask_batch) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch)

            # Calculate masked binary cross-entropy loss
            loss = masked_bce_loss(y_pred, y_batch, mask_batch)  # Pass mask_batch, not mask_tensor
            total_loss += loss.item()  # Accumulate batch loss

            # Backward pass
            loss.backward()
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

    # Return the trained model
    return model, epoch_losses