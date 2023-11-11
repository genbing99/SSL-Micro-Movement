import torch
from torch.utils.data import Dataset

# Convert dataset into tensor for training and testing 
class OFFSpottingDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        return x, y