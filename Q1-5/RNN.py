import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from data_prep import pad_and_convert, load_imdb
import itertools


class CustomSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes=2):
        super(CustomSeq2SeqModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # RNN layer (Elman network)
        self.elman_rnn = nn.RNN(input_size=emb_size, hidden_size=hidden_size, batch_first=True)

        # Output layer to project down to the number of classes
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        # Input sequence shape: (batch, time)
        embedded = self.embedding(input_seq)  # (batch, time, emb)

        # Elman network (RNN) layer
        elman_output, _ = self.elman_rnn(embedded)

        # Global max pool along the time dimension for Elman network
        max_pooled = torch.max(elman_output, dim=1)[0]  # (batch, hidden)

        # Project down to the number of classes
        output = self.output_layer(max_pooled)  # (batch, num_classes)

        return output

# Load data and prepare batches
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train = pad_and_convert(x_train)
x_val = pad_and_convert(x_val)

batch_size = 64

# ----------------TRAINING SET------------------
# Convert x_train and y_train to PyTorch tensors
x_train_tensor = torch.tensor(x_train).clone().detach()
y_train_tensor = torch.tensor(y_train, dtype=torch.long).clone().detach()

# Create a TensorDataset from x_train and y_train
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Create a DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------------VALIDATION SET------------------
# Convert x_val and y_val to PyTorch tensors
x_val_tensor = torch.tensor(x_val).clone().detach()
y_val_tensor = torch.tensor(y_val, dtype=torch.long).clone().detach()

# Create a TensorDataset from x_val and y_val
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Create a DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#---------------------------------------------------

vocab_size = len(set(i2w))

# HYPERARAMETERS
param_grid = {
    'learning_rate': [0.005, 0.001, 0.0005],
    'hidden_size': [300, 500],
    'emb_size': [300, 500],
    'epochs': [1, 3], 
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(*param_grid.values()))

best_accuracy = 0

# Iterate over each combination
for params in param_combinations:
    # Extract hyperparameter values
    learning_rate, hidden_size, emb_size, epochs = params

    print(f"Training with hyperparameters: lr={learning_rate}, hidden_size={hidden_size}, emb_size={emb_size}, epochs={epochs}")

    # Create an instance of the model with the current hyperparameters
    model = CustomSeq2SeqModel(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, num_classes=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = epochs

    # -----------------TRAINING-----------------
    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            inputs, labels = batch

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        # Validation (optional)
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0

            for val_batch in val_loader:
                val_inputs, val_labels = val_batch
                val_outputs = model(val_inputs)
                predicted_labels = torch.argmax(val_outputs, dim=1)
                total_correct += (predicted_labels == val_labels).sum().item()
                total_samples += val_labels.size(0)

            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        

    print("===================================")

# Print best accuracy and parameters
print(f"Best accuracy: {best_accuracy}")
print(f"Best parameters: lr={best_params[0]}, hidden_size={best_params[2]}, emb_size={best_params[3]}, epochs={best_params[4]}")

# Save best model params
torch.save(best_params, "best_params_RNN.pth")