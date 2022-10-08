import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data


# use 0 to padding


class TrainDataset(Data.Dataset):
    def __init__(self, max_len, path, device):
        self.data = pd.read_csv(path, header=None).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, -self.max_len - 3:-3].tolist()
        pos = self.data[index, -self.max_len - 2:-2].tolist()
        pos = pos[-len(seq):]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        padding_len = self.max_len - len(pos)
        pos = [0] * padding_len + pos

        return torch.LongTensor(seq).to(self.device), torch.LongTensor(pos).to(self.device)


class EvalDataset(Data.Dataset):
    def __init__(self, max_len, mode, path, device):
        self.data = pd.read_csv(path, header=None).values if mode == 'test' else pd.read_csv(path, header=None).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = self.num_item + 1
        self.max_len = max_len
        self.mode = mode
        self.device = device

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index, -2] if self.mode == 'val' else self.data[index, -1]

        seq = list(seq)
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        answers = [pos]
        return torch.LongTensor(seq).to(self.device), torch.LongTensor(answers).to(self.device)

