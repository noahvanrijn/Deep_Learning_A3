import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from data_prep import pad_and_convert, load_imdb

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
        max_pooled = torch.max(relu_output, dim=1)[0]  # (batch, hidden)

        # Project down to the number of classes
        output = self.output_layer(max_pooled)  # (batch, num_classes)

        return output

# Load data and prepare batches
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train = pad_and_convert(x_train)
x_val = pad_and_convert(x_val)

# ----------------TRAINING SET------------------
# Convert x_train and y_train to PyTorch tensors
x_train_tensor = torch.tensor(x_train).clone().detach()
y_train_tensor = torch.tensor(y_train, dtype=torch.long).clone().detach()

# Create a TensorDataset from x_train and y_train
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Create a DataLoader for the training set
batch_size = 32  # You can adjust the batch size as needed
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

# Create an instance of the model
model = CustomSeq2SeqModel(vocab_size=vocab_size, emb_size=300, hidden_size=300, num_classes=2)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1


#-----------------TRAINING-----------------
for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        inputs, labels = batch

        print(inputs.shape)
        print(inputs)

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
