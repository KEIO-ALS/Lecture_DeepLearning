import torch
from torch.utils.data import Dataset
import os

# step 6
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, input_dir) -> None:
        # データディレクトリへのパスを作成
        self.input_dir = input_dir
        case_dir = os.path.join(input_dir, "case")
        control_dir = os.path.join(input_dir, "control")
        # ファイル名一覧を取得
        self.case_file_names = os.listdir(case_dir)
        self.control_file_names = os.listdir(control_dir)
        # 総データ数を取得
        self.num_files = len(self.case_file_names + self.control_file_names)
    
    def __getitem__(self, index):
        if index < len(self.case_file_names):
            filename = self.case_file_names[index]
            filename = f"case/{filename}"
            label = 1
        else:
            filename = self.control_file_names[index-len(self.case_file_names)]
            filename = f"control/{filename}"
            label = 0
        file_path = os.path.join(self.input_dir, filename)
        data = torch.load(file_path)
        return data, label
    
    def __len__(self):
        return self.num_files


class ComplexDataset(Dataset):
    def __init__(self, input_dir):
        self.data = np.load(os.path.join(input_dir, "data.npy"))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = np.load(os.path.join(input_dir, "label.npy"))
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index].long()