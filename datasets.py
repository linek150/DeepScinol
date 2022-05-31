from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, name: str):
        self.name = name
        if name == '1dLinear':
            self.x = torch.arange(-5, 5, 0.1).view(-1, 1)
            self.y = -5 * self.x + 0.1 * torch.randn(self.x.size())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.name == '1dLinear':
            return self.x[idx], self.y[idx]