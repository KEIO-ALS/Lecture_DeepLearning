
# 課題：「健康診断の結果から、糖尿病を診断したい」

# 目標：深層学習モデルを動かしてみる
# 準備：models/perceptron.pyを作る


from models import Perceptron
import torch


# 入力データ(x)を作成(今回はランダム)
# データは、バッチサイズ2 / 次元数4　と仮定
# 　　　　　身長　体重　視力　 血圧
# Aさん　　 175  70   1.00  125
# Bさん　　 165　95　　0.01　170
# みたいな
x = torch.randn(size=(2,4))

# モデルをインスタンス化
model = Perceptron(
    # 入力するデータの次元数
    input_dim=4,
    # 出力するデータの次元数
    output_dim=2,
)

# モデルにデータを入力し、出力を得る
output = model(x)

# 出力を見てみる
# バッチサイズ2 / 次元数2　が出力されるはず！
# 　　　　　糖尿病率　非糖尿病率
# Aさん　　 0.1　　　　0.9
# Bさん　　 0.9       0.1
# というイメージ
print(output)