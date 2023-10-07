

# 課題：「健康診断の結果から、糖尿病を診断したい」

# 目標：学習を評価する
# 準備：なし


from models import Perceptron, CustomDataset, plot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]

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


record = {
    "loss/step":[],
    "loss/epoch":[],
    "accuracies":[],
}

num_epochs = 10
for e in range(1,num_epochs+1):
    epoch_loss = 0
    num_batch = 0
    print(f"start epoch {e}!")
    for data, label in tqdm(dataloader):
        # トレーニング
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        # 正解数の計算
        _, predicted = torch.max(outputs.data, 1)
        total = label.size(0)
        correct = (predicted == label).sum().item()
        # 正解率をレコーディング
        record["accuracies"].append(correct/total)
        
        # ステップ損失をレコーディング
        record["loss/step"].append(loss.item())
        epoch_loss += loss.item()
        num_batch += 1
    
    # エポック損失をレコーディング
    avg_epoch_loss = epoch_loss / num_batch
    record["loss/epoch"].append(avg_epoch_loss)

# 損失/精度の描画
plot(output_dir, "loss_per_step", 0.05, record["loss/step"])
plot(output_dir, "loss_per_epoch", 0.5, record["loss/epoch"])
plot(output_dir, "accuracies", 0.05, record["accuracies"])