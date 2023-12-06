from data_rnn import load_imdb, gen_sentence
import numpy as np
import torch 

# Load the data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

# Print a sentence
print([i2w[w] for w in x_train[141]])
print(x_train[141])

# Function to pad and convert the batch
def pad_and_convert(batch):
    max_len = max([len(x) for x in batch])
    batch = [x + [w2i['.pad']] * (max_len - len(x)) for x in batch]
    batch = torch.tensor(batch, dtype=torch.long)
    return batch

# Convert the batch
train = pad_and_convert(x_train)
print([i2w[w] for w in train[141]])
print(train[141])



