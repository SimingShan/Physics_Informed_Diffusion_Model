import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VorticityDataset(Dataset):
    def __init__(self, numpy_data, transform=None):
        self.data = numpy_data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

def load_data(file_path):
    hi_res = np.load(file_path)
    train_data = hi_res[:32].reshape(-1, hi_res.shape[2], hi_res.shape[3])
    dev_data = hi_res[32:36]
    test_data = hi_res[36:]
    return train_data, dev_data, test_data

def prepare_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device