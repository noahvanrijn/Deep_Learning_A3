import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
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
print(x_train[0])
print(y_train[0])

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
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    model.train()

    for train in train_loader:
        inputs, labels = train
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
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
