from data_rnn import load_brackets, load_brackets
# from data_prep import pad_and_convert
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist
from torch.optim.lr_scheduler import ExponentialLR
from itertools import product
import random


x_train_brackets, (i2w_brackets, w2i_brackets) = load_brackets(n=150_000)

vocab_size = len(w2i_brackets)
num_char = vocab_size
num_epochs = 20
max_length = 10
patience = 3
num_trials = 20


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_char, n_layers=1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_char)
        self.batch_norm = nn.BatchNorm1d(num_char)
    
    def forward(self, input_seq, h=None):
        embedded = self.embedding(input_seq)
        b, t, e = embedded.size()
        lstm_out, hidden = self.lstm(embedded, h)

        output = self.fc(lstm_out)
        output = output.permute(0, 2, 1)
        output = self.batch_norm(output)
        output = output.permute(0, 2, 1)

        return output, hidden

def pad_and_convert3(batch, w2i, batch_size):
    start_token = w2i['.start']
    end_token = w2i['.end']
    pad_token = w2i['.pad']

    num_examples = len(batch)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    batches = [batch[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    
    padded_batches = []
    
    for b in batches:
        b = [[start_token] + x + [end_token] for x in b]
        max_len = max(len(x) for x in b)
        padded_batch = [x + [pad_token] * (max_len - len(x)) for x in b]
        padded_batches.append(padded_batch)

    padded_batches = [torch.tensor(pb, dtype=torch.long) for pb in padded_batches]

    return padded_batches

def create_target2(padded_batches):
    target_batches = []

    for batch in padded_batches:
        target_batch = torch.cat((batch[:, 1:], torch.zeros(batch.size(0), 1).int()), dim=1)
        target_batches.append(target_batch)
    return target_batches



def sample(lnprobs, temperature=1.0): 
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given
        distribution, 0.0 returns the maximum probability element. :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


search_space = {
    'learning_rate': [0.0005, 0.001, 0.01, 0.1],
    'emb_size': [32, 64, 128, 356, 512],
    'hidden_size': [32, 64, 128, 256, 512],
    'n_layers': [1, 2, 3],
    'batch_size': [16, 32, 64, 128, 256]
}


for trial in range(num_trials):

    no_improvement_count = 0    
    best_loss = float('inf')
    # Randomly sample hyperparameters for this trial
    learning_rate = random.choice(search_space['learning_rate'])
    emb_size = random.choice(search_space['emb_size'])
    hidden_size = random.choice(search_space['hidden_size'])
    n_layers = random.choice(search_space['n_layers'])
    batch_size = random.choice(search_space['batch_size'])
    print(f"Hyperparameters: lr={learning_rate}, emb_size={emb_size}, hidden_size={hidden_size}, n_layers={n_layers}, batch_size={batch_size}")

    x_train_brackets_padded3 = pad_and_convert3(x_train_brackets, w2i_brackets, batch_size)
    target_brackets2 = create_target2(x_train_brackets_padded3)
    trainloader_brackets = list(zip(x_train_brackets_padded3, target_brackets2))

    model = LSTM(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, num_char=num_char, n_layers=n_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader_brackets):

            model.train()        
            optimizer.zero_grad()

            h = None
            
            output, _ = model(inputs, h)
            output = output.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(output, targets)  

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # model.eval()
        # with torch.no_grad():
        #     print(f'Generated samples after epoch {epoch+1}')
        #     for _ in range(10):
        #         seed_seq = [w2i_brackets['.start'], w2i_brackets['('], w2i_brackets['('], w2i_brackets[')']]
        #         generated_sequence = []
        #         for t in range(max_length - 1):
        #             seed_input = torch.tensor([seed_seq], dtype=torch.long)
        #             output, _ = model(seed_input, h)
        #             next_token = sample(output[0, -1, :], temperature=0.5)

        #             if next_token == w2i_brackets['.end']:
        #                 break

        #             seed_seq.append(next_token.item())
        #             seed_input = torch.tensor([[next_token]], dtype=torch.long)
                
        #         seed_seq.remove(w2i_brackets['.start'])
        #         generated_sequence = ''.join(i2w_brackets[i] for i in seed_seq)
        #         print(f'{generated_sequence}')
        
        # scheduler.step()

        average_loss = total_loss / len(trainloader_brackets)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

        # Early stopping
        if average_loss < best_loss:
            best_loss = average_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Early stopping at epoch {epoch+1} as there has been no improvement for {patience} consecutive epochs.')
            break

    torch.save(model.state_dict(), 'lstm_model_brackets.pth')

