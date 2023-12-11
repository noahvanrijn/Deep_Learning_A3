import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from data_prep import pad_and_convert, load_imdb


class Elman(nn.Module):
    def __init__(self, vocab_size, emb_size=300, insize=300, outsize=300, hsize=300):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.lin1 = torch.nn.Linear(insize + hsize, hsize)
        self.lin2 = torch.nn.Linear(hsize, outsize)

    def forward(self, x, hidden=None):

        b, t, e = x.size()

        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)
        outs = []

        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            #...
            hidden = F.relu(self.lin1(inp))
            out = self.lin2(hidden)

            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden
    

