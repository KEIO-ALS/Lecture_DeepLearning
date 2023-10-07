

# 課題：「健康診断の結果から、糖尿病を診断したい」

# 目標：学習を評価する
# 準備：なし


from models import NeuralNetwork, CustomDataset, plot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys

from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]

dataset = CustomDataset(input_dir)

split_rate = 0.8
n_train = int(len(dataset) * split_rate)
n_test   = int(len(dataset) - n_train)
train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=50,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=50,
    shuffle=False,
)


model = NeuralNetwork(
    input_dim=4,
    hidden_dims=[10,10,10],
    output_dim=2
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

record = {
    "loss/step":[],
    "loss/epoch":[],
    "train_accuracies":[],
    "test_accuracies":[],
}
num_epochs = 10
for e in range(1,num_epochs+1):
    epoch_loss = 0
    num_batch = 0
    total = 0
    correct = 0
    print(f"start epoch {e}!")
    for data, label in tqdm(train_dataloader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        record["loss/step"].append(loss.item())
        epoch_loss += loss.item()
        num_batch += 1
    record["train_accuracies"].append(correct/total)
    avg_epoch_loss = epoch_loss / num_batch
    record["loss/epoch"].append(avg_epoch_loss)
    
    with torch.no_grad():
        total = 0
        correct = 0
        for data, label in tqdm(test_dataloader):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        record["test_accuracies"].append(correct/total)
            

plot(output_dir, "loss_per_step", 0.05, record["loss/step"])
plot(output_dir, "loss_per_epoch", 0.5, record["loss/epoch"])
plot(output_dir, "accuracies", -1, record["train_accuracies"], record["test_accuracies"])