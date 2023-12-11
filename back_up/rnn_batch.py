import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from data_prep import pad_and_convert, load_imdb
import random
from torch.utils.data import DataLoader, TensorDataset


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
        embedded = self.embedding(torch.stack(input_seq, dim=1))  # (batch, time, emb)

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
#train_loader = list(zip(x_train, y_train))

#-----------------new code-----------------
# Convert x_train and y_train to PyTorch tensors
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Create a TensorDataset from x_train and y_train
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Create a DataLoader for the training set
batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#-----------------new code-----------------

val_loader = list(zip(x_val, y_val))

vocab_size = len(set(i2w))

# Create an instance of the model
model = CustomSeq2SeqModel(vocab_size=vocab_size, emb_size=300, hidden_size=300, num_classes=2)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 32  

for epoch in range(num_epochs):
    model.train()

    # Shuffle the training data for each epoch
    random.shuffle(train_loader)

    for i in range(0, len(train_loader), batch_size):
        batch = train_loader[i:i+batch_size]
        
        # Unzip the batch
        inputs, labels = zip(*batch)

        print(len(inputs))
        print(inputs)
        
        # Change labels from int to tensor
        labels = torch.tensor(labels, dtype=torch.long)

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

        for val in val_loader:
            inputs, labels = val
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += 1

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
