import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import GD
from data_prep import pad_and_convert, load_imdb

class Elman(nn.Module):
    def __init__(self, vocab_size, emb_size=300, insize=300, outsize=300, hsize=300):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.lin1 = torch.nn.Linear(insize + hsize, hsize)
        self.lin2 = torch.nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):
        print(x)
        x = self.embedding(x)
        print(x.size())
        b, t, e = x.size()

        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)
        outs = []

        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            print(inp)
            print(inp.size())
            #...
            hidden = F.relu(self.lin1(inp))
            out = self.lin2(hidden)

            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden
    

#--------------------------------------------
# Load data and prepare batches
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train = pad_and_convert(x_train)
x_val = pad_and_convert(x_val)

# pair the x_train and y_train
train_loader = list(zip(x_train, y_train))
val_loader = list(zip(x_val, y_val))

vocab_size = len(set(i2w))

# Create an instance of the model
model = Elman(vocab_size=vocab_size, emb_size=300, insize=300, outsize=300, hsize=300)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = GD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 32

for epoch in range(num_epochs):
    model.train()

    #for train in train_loader:
    for train in range(0, len(train_loader), batch_size):
        inputs, labels = train

        # Change label from int to tensor
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


