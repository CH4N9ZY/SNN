#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from torch.utils.data import Dataset, DataLoader
import re
import os

class DVS_CIFAR10(Dataset):
    
    def __init__(self, dataPath, mode):#dataPath = '../../../DATA/DVS-CIFAR10/processed/'
        self.mode = mode
        self.dataPath = dataPath + 'tensor_' + self.mode + '/'
        self.dataFiles = os.listdir(self.dataPath)#self.dataPath = '../../../DATA/DVS-CIFAR10/processed/tensor_train/'
    
    def __getitem__(self, index):
        tensor_file = self.dataFiles[index]
        tensor_data = torch.load(self.dataPath + tensor_file)
        
        label_pattern = r'(\d+)_'
        label = torch.tensor(int(re.findall(label_pattern, tensor_file)[0]), dtype = torch.int64)
        
        return tensor_data, label
    
    def __len__(self):
        return len(self.dataFiles)

