import torch
import torch.nn as nn
import torch.nn.functional as F

class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.lin1 = ...
        self.lin2 = ...


    def forward(self, x, hidden=None):
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float)
        outs = []
        
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            ...
            outs.append(out[:, None, :])
        return torch.cat(outs, dim=1), hidden