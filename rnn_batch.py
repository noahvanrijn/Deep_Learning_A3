import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import GD
from torch.utils.data import DataLoader
from data_prep import pad_and_convert, load_imdb
import random


class CustomSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, emb_size=300, hidden_size=300, num_classes=2):
        super(CustomSeq2SeqModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # Linear layer with ReLU activation and global max pool
        self.shared_mlp = nn.Linear(emb_size, hidden_size)

        # Output layer to project down to the number of classes
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        # Input sequence shape: (batch, time)
        embedded = self.embedding(input_seq)  # (batch, time, emb)

        # Map each token by a shared MLP
        mlp_output = self.shared_mlp(embedded)  # (batch, time, hidden)

        # Apply ReLU activation
        relu_output = F.relu(mlp_output)

        # Global max pool along the time dimension
        max_pooled = torch.max(relu_output, dim=0)[0]  # (batch, hidden)

        # Project down to the number of classes
        output = self.output_layer(max_pooled)  # (batch, num_classes)

        return output

# Load data and prepare batches
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train = pad_and_convert(x_train)
x_val = pad_and_convert(x_val)

# pair the x_train and y_train
train_loader = list(zip(x_train, y_train))
val_loader = list(zip(x_val, y_val))

vocab_size = len(set(i2w))

# Create an instance of the model
model = CustomSeq2SeqModel(vocab_size=vocab_size, emb_size=300, hidden_size=300, num_classes=2)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = GD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 32

for epoch in range(num_epochs):
    model.train()

    # Shuffle your training data for each epoch
    random.shuffle(train_loader)

    for batch_start in range(0, len(train_loader), batch_size):
        batch_end = min(batch_start + batch_size, len(train_loader))
        batch_data = train_loader[batch_start:batch_end]

        batch_inputs, batch_labels = zip(*batch_data)

        # Convert labels to tensors
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        optimizer.zero_grad()  # Zero the gradients

        # Concatenate inputs into a single tensor
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)
        
        # Forward pass
        outputs, _ = model(batch_inputs)

        # Calculate loss
        loss = criterion(outputs, batch_labels)

        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()

    # Validation (optional)
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for val_start in range(0, len(val_loader), batch_size):
            val_end = min(val_start + batch_size, len(val_loader))
            val_batch = val_loader[val_start:val_end]

            val_inputs, val_labels = zip(*val_batch)

            # Convert labels to tensors
            val_labels = torch.tensor(val_labels, dtype=torch.long)

            # Concatenate inputs into a single tensor
            val_inputs = torch.tensor(val_inputs, dtype=torch.long)

            # Forward pass
            val_outputs, _ = model(val_inputs)

            # Calculate accuracy
            predicted_labels = torch.argmax(val_outputs, dim=1)
            total_correct += (predicted_labels == val_labels).sum().item()
            total_samples += len(val_batch)

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
