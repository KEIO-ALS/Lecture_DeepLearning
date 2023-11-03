from models import NeuralNetwork, Recurrent

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

batch_size = 16
seq_len = 8
input_dim = 10
num_train_data = 100
num_test_data = 10

hidden_dims = [20, 30]
output_dim = 10

learning_rate = 0.001
num_epochs = 20

# random data
class random_dataset(Dataset):
    def __init__(self, num_data, seq_len, input_dim, output_dim):
        self.data, self.label = [], []
        for _ in range(num_data):
            self.data.append(torch.randn(seq_len, input_dim))
            self.label.append(torch.randn(seq_len, output_dim))
            
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
        
train_dataset = random_dataset(num_train_data, seq_len, input_dim, output_dim)
test_dataset = random_dataset(num_test_data, seq_len, input_dim, output_dim)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# settings
net = NeuralNetwork(input_dim+output_dim, hidden_dims, output_dim)
model = Recurrent(net)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    for data, label in tqdm(train_dataloader):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# evaluate
model.eval()
with torch.no_grad():
    count = 0
    loss = 0.
    for data, label in tqdm(test_dataloader):
        outputs, _ = model(data)
        loss += criterion(outputs, label).item()
        count += 1
    print(f'Test Loss: {loss/count:.4f}')