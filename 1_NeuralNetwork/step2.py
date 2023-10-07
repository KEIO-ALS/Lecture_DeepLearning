

# 課題：「健康診断の結果から、糖尿病を診断したい」

# 目標：深層学習モデルをランダムデータで学習させてみる
# 準備：なし


from models import Perceptron

import torch
import torch.nn as nn
import torch.optim as optim


# 入力データ(x)を作成(今回はランダム)
# データは、バッチサイズ2 / 次元数4　と仮定
# 　　　　　身長　体重　視力　 血圧
# Aさん　　 175  70   1.00  125
# Bさん　　 165　95　　0.01　170
# みたいな
x = torch.randn(size=(2,4))


# ラベルデータ(x)を作成(今回はランダム)
# データは、バッチサイズ2 / 次元数2　と仮定
# 　　　　　糖尿病 糖尿病でない
# Aさん　　  0     1
# Bさん　　  1     0
# みたいな
y = torch.randn(size=(2,2))

# モデルをインスタンス化
model = Perceptron(
    # 入力するデータの次元数
    input_dim=4,
    # 出力するデータの次元数
    output_dim=2,
)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

for _ in range(5):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"loss: {loss.item()}")