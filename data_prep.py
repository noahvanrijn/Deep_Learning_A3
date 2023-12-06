from data_rnn import load_imdb, gen_sentence
import numpy as np
import torch 

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

print([i2w[w] for w in x_train[141]])
print(x_train[141])

print(max([len(x) for x in x_train]))

print([i2w[w] for w in x_train[2514]])


# To train, you'll need to loop over x_train and y_train and slice out batches. Each batch
# will need to be padded to a fixed length and then converted to a torch tensor.
# question 1: Implement this padding and conversion. Show the function in your report.

# Tips:
# ● We've included a special padding token in the vocabulary, represented by the string ".pad". Consult the w2i dictionary to see what the index of this token is.
# ● We've also included special tokens ".start" and ".end", which are only used in the autoregressive task.
# ● If you feed a list of lists to the function torch.tensor(), it'll return a torch tensor.
# ○ The inner lists must all have the same size
# ○ Pytorch is pretty good at guessing which datatype (int, float, byte) is
# expected, but it does sometimes get it wrong. To be sure, add the dataype with batch = torch.tensor(lists, dtype=torch.long).

def pad_and_convert(batch):
    max_len = max([len(x) for x in batch])
    batch = [x + [w2i['.pad']] * (max_len - len(x)) for x in batch]
    batch = torch.tensor(batch, dtype=torch.long)
    return batch

train = pad_and_convert(x_train)
print([i2w[w] for w in train[141]])
print(train[141])



