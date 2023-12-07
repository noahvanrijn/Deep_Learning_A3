from data_rnn import load_imdb, gen_sentence
import numpy as np
import torch 

# Load the data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

# Function to pad and convert the batch
def pad_and_convert(batch):
    max_len = max([len(x) for x in batch])
    batch = [x + [w2i['.pad']] * (max_len - len(x)) for x in batch]
    batch = torch.tensor(batch, dtype=torch.long)
    return batch

print(x_train[0])
print(y_train[0])

