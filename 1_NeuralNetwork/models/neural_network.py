import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # hidden layers
        modules = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            modules.extend([nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()])
        
        # output layer
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)