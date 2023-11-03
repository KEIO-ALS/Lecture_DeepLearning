from .NN import NeuralNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F


class Recurrent(nn.Module):
    def __init__(self, net):
        super().__init__()
        
        self.net = net
        self.hidden_dim = net.output_dim

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        outputs = []
        for i in range(seq_len):
            xi = x[:, i, :]
            combined = torch.cat((xi, hidden), dim=1)
            hidden = self.net(combined)
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)   
        return outputs, hidden