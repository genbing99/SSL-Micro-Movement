import glob
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from aifc import Error

class OFFDataset(Dataset):
    def __init__(self, root, mode="train"):
        self.mode = mode
        self.files = sorted(glob.glob("%s/*/*.npy" % root))
        self.files = random.sample(self.files, 2000) if mode == "val" else random.sample(self.files, 100000)

    def __len__(self):
        return len(self.files)

    def getLabel(self, distance):
        # {1,5,10} is the default settings
        if   distance == 1:
            return 0
        elif distance == 5:
            return 1
        elif distance == 10:
            return 2
        raise Error

    def __getitem__(self, index):
        filename = self.files[index % len(self.files)]
        img = np.load(filename)

        # Convert to tensor
        # Example of filename: 15_0101disgustingteeth/000001_10.npy
        img = torch.as_tensor(img).permute(2,0,1)
        distance = int(filename.split('\\')[-1].split('.')[0].split('_')[1])
        label = self.getLabel(distance)
        return img, label

        