from models import Perceptron

import torch

model = Perceptron(
    input_dim=5,
    output_dim=2,
)

x = torch.randn(size=(2,5))

output = model(x)

print(output.shape)