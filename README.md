# 初学者向け深層学習教材

## 準備

### 環境構築１ (初回のみ)

1．仮想環境の作成
```
python -m venv venv_DL
cd venv_DL
```

1. gitリポジトリの獲得
```
git clone https://github.com/KEIO-ALS/Lecture_DeepLearning.git
```


### 仮想環境の起動 (毎回)

1. 仮想環境を作動させる
必要であれば`cd path/to/venv_DL`で先にディレクトリに移動
- Mac
    ```
    . bin/activate
    cd Lecture_DeepLearning
    ```
- Windows
    ```
    Scripts\activate
    cd Lecture_DeepLearning
    ```


### 環境構築２ (初回のみ)

1. 外部ライブラリ
```
python -m pip install -U pip
pip install -r requirements.txt
```

1. データセット
データセットを[GoogleDrive](https://drive.google.com/file/d/1NOmv4nSnx9cnPUzORlZtfCDZ1pdeuED3/view?usp=sharing)からダウンロードし、`1_NeuralNetwork`ディレクトリに解凍


## 全結合

## 自己回帰モデル

## 強化学習
https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

1. 学習済みモデルの実行
```
cd path/to/3_ReinforcementLearning
```
2000エピソード学習したモデル
```
python sample.py 2K
```
10000エピソード学習したモデル
```
python sample.py 10K
```