from data_rnn import load_ndfa, load_brackets
# from data_prep import pad_and_convert
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributions as dist

x_train_ndfa, (i2w_ndfa, w2i_ndfa) = load_ndfa(n=150_000)

vocab_size = len(w2i_ndfa)
emb_size = 32
h = 16
num_char = vocab_size
n_layers = 1
num_epochs = 3
learning_rate = 0.001
max_length = 50

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, h, num_char, n_layers=1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=h, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(h, num_char)
    
    def forward(self, input_seq, h=None):
        embedded = self.embedding(input_seq)
        lstm_out, hidden = self.lstm(embedded, h)
        # lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        # raise Exception('stop')
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

x_train_ndfa_padded3 = pad_and_convert3(x_train_ndfa, w2i_ndfa, batch_size=64)
target_ndfa2 = create_target2(x_train_ndfa_padded3)
trainloader_ndfa = list(zip(x_train_ndfa_padded3, target_ndfa2))


model = LSTM(vocab_size=vocab_size, emb_size=emb_size, h=h, num_char=num_char, n_layers=1)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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


for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader_ndfa):

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

    model.eval()
    with torch.no_grad():
        print(f'Generated samples after epoch {epoch+1}')
        for _ in range(10):
            seed_seq = [w2i_ndfa['.start'], w2i_ndfa['v'], w2i_ndfa['k'], w2i_ndfa['a']]
            generated_sequence = []

            for t in range(max_length - 1):
                seed_input = torch.tensor([seed_seq], dtype=torch.long)
                output, _ = model(seed_input, h)
                # print('output', output[0, -1, :])
                next_token = sample(output[0, -1, :], temperature=1.0)
                # print('next token', next_token)
                

                if next_token == w2i_ndfa['.end']:
                    break

                seed_seq.append(next_token.item())
                seed_input = torch.tensor([[next_token]], dtype=torch.long)
                # print('seed seq', seed_seq)
            seed_seq.remove(w2i_ndfa['.start'])
            generated_sequence = ''.join(i2w_ndfa[i] for i in seed_seq)
            print(f'{generated_sequence}')

    average_loss = total_loss / len(trainloader_ndfa)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

torch.save(model.state_dict(), 'lstm_model_ndfa.pth')