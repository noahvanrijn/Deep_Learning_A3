import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Assuming you have a function named 'load_imdb' that returns the vocabulary size
# Replace 'vocab_size' with the actual vocabulary size from your 'load_imdb' function
vocab_size = load_imdb()[0]

# Create an instance of the model
model = CustomSeq2SeqModel(vocab_size=vocab_size, emb_size=300, hidden_size=300, num_classes=2)
