from data_rnn import load_ndfa, load_brackets
# from data_prep import pad_and_convert
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from itertools import product
import random


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_char, n_layers=1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_char)

        self.init_weights()
        # self.batch_norm = nn.BatchNorm1d(num_char)
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

    def forward(self, input_seq, h=None):
        embedded = self.embedding(input_seq)
        lstm_out, hidden = self.lstm(embedded, h)
        # lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        # output = output.permute(0, 2, 1)
        # output = self.batch_norm(output)
        # output = output.permute(0, 2, 1)
        # raise Exception('stop')
        return output, hidden

def create_target2(padded_batches):
    target_batches = []

    for batch in padded_batches:
        target_batch = torch.cat((batch[:, 1:], torch.zeros(batch.size(0), 1).int()), dim=1)
        target_batches.append(target_batch)
    return target_batches

def pad_and_convert3(batch, w2i, batch_size):
    start_token = w2i['.start']
    end_token = w2i['.end']
    pad_token = w2i['.pad']

    num_examples = len(batch)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    # Create batches
    batches = [batch[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    
    padded_batches = []
    
    for b in batches:
        # Pad each sequence in the batch to the maximum length within the batch
        b = [[start_token] + x + [end_token] for x in b]
        max_len = max(len(x) for x in b)
        padded_batch = [x + [pad_token] * (max_len - len(x)) for x in b]
        padded_batches.append(padded_batch)

    # Convert the padded batches to PyTorch tensors
    padded_batches = [torch.tensor(pb, dtype=torch.long) for pb in padded_batches]

    return padded_batches



x_train_ndfa, (i2w_ndfa, w2i_ndfa) = load_ndfa(n=150_000)

vocab_size = len(w2i_ndfa)
num_char = vocab_size
patience = 5

search_space = {
    'learning_rate': [0.1],
    'emb_size': [32],
    'hidden_size': [16],
    'n_layers': [1],
    'batch_size': [64]
}



num_trials = 1
num_epochs = 50

for trial in range(num_trials):
    print(f"Trial {trial + 1}/{num_trials}")
    no_improvement_count = 0
    best_val_loss = float('inf')

    # Randomly sample hyperparameters for this trial
    learning_rate = random.choice(search_space['learning_rate'])
    emb_size = random.choice(search_space['emb_size'])
    hidden_size = random.choice(search_space['hidden_size'])
    n_layers = random.choice(search_space['n_layers'])
    batch_size = random.choice(search_space['batch_size'])
    print(f"Hyperparameters: lr={learning_rate}, emb_size={emb_size}, hidden_size={hidden_size}, n_layers={n_layers}, batch_size={batch_size}")

    # Instantiate the model with the sampled hyperparameters
    model = LSTM(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, num_char=num_char, n_layers=n_layers)

    # Prepare the data with the current batch_size
    x_train_ndfa_padded3 = pad_and_convert3(x_train_ndfa, w2i_ndfa, batch_size)
    target_ndfa2 = create_target2(x_train_ndfa_padded3)
    x_train, x_val, y_train, y_val = train_test_split(x_train_ndfa_padded3, target_ndfa2, test_size=0.2, random_state=42)

    trainloader = list(zip(x_train, y_train))
    valloader = list(zip(x_val, y_val))

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)


    for epoch in range(num_epochs):

        # Training loop
        total_loss = 0.0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()

            h = None
            output, _ = model(inputs, h)

            output = output.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(output, targets)
            total_loss += loss.item()
            non_pad_tokens = torch.sum(targets != 0).item()
            total_tokens += non_pad_tokens
            loss /= non_pad_tokens

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        average_loss = total_loss / total_tokens
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for inputs, targets in valloader:
                output, _ = model(inputs, h)
                output = output.reshape(-1, vocab_size)
                targets = targets.reshape(-1)
                loss = criterion(output, targets)
                val_loss += loss.item()
                non_pad_tokens = torch.sum(targets != 0).item()
                val_tokens += non_pad_tokens

        average_val_loss = val_loss / val_tokens

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

        # Early stopping
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch+1} as there has been no improvement for {patience} consecutive epochs.')
            break

# for epoch in range(num_epochs):
# total_loss = 0.0
# total_tokens = 0

#     for batch_idx, (inputs, targets) in enumerate(trainloader):

#         model.train()        
#         optimizer.zero_grad()

#         h = None
        
#         output, _ = model(inputs, h)

#         output = output.reshape(-1, vocab_size)
#         targets = targets.reshape(-1)


#         loss = criterion(output, targets)  

#         total_loss += loss.item()
#         non_pad_tokens = torch.sum(targets != 0).item()
#         total_tokens += non_pad_tokens
#         loss /= non_pad_tokens

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

#         optimizer.step()

#     average_loss = total_loss / total_tokens

#     print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')
    