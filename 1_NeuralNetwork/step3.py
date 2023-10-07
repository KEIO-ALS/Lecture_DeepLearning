

# 課題：「健康診断の結果から、糖尿病を診断したい」

# 目標：深層学習モデルをサンプルデータで学習させてみる
# 準備：models/custom_dataset.pyを作る

from models import Perceptron, CustomDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

input_dir = sys.argv[1]

dataset = CustomDataset(input_dir)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=50,
    shuffle=True,
)

model = Perceptron(
    input_dim=4,
    output_dim=2,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for e in range(3):
    print(f"start epoch {e}!")
    for data, label in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")