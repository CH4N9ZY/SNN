# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:12:29 2021

@author: 常子翼
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from IterativeLIFModel import *
from SCNNLayer import *
import time

class DVS_Gesture(Dataset):
    
    def __init__(self, dataPath, mode, duration, timestep):
        
        self.mode = mode
        if mode == 'train':
            self.dataPath = dataPath + 'tensor_train/' #../../../DATA/DVS-Gesture/processed/dvs_train/
            self.dataFiles = os.listdir(self.dataPath) #a list
        elif mode == 'test':
            self.dataPath = dataPath + 'tensor_test/' #../../../DATA/DVS-Gesture/processed/dvs_test
            self.dataFiles = os.listdir(self.dataPath)
        else:
            print('typo!')
        #self.duration = duration # 30*1000 #ms->um，每次采样持续30毫秒(30*1000微秒)
        #self.timestep = timestep # 1200 * 1000 #ms->um，整个采样时长
        #self.bins = int(self.timestep/self.duration) # 次，每个动作采样40次
    
    def __getitem__(self, index):
        data_dir = self.dataPath + self.dataFiles[index]
        tensor_data = torch.load(data_dir)
        label_pattern = r'_(\d+)_'
        label = int(re.findall(label_pattern,self.dataFiles[index])[0]) - 1
        label  = torch.tensor(label, dtype=torch.int64)
        return tensor_data, label
    
    def __len__(self):
        return len(self.dataFiles)


dataPath = '../../../DATA/DVS-Gesture/processed/'
duration = 30*1000
timestep = 1200 * 1000

trainSet = DVS_Gesture(dataPath = dataPath, mode = 'train')
trainLoader = DataLoader(dataset = trainSet, batch_size=8, shuffle=True, num_workers = 0)

testSet = DVS_Gesture(dataPath = dataPath, mode = 'test')
testLoader = DataLoader(dataset = testSet, batch_size=8, shuffle=True, num_workers = 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = []
num_epochs = 100 # max epoch
num_classes = 11
batch_size = 8
names = 'STBPmodelDVS-Gesture'

learning_rate = 1e-3

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

for epoch in range(num_epochs):
    train_loss = 0
    start_time = time.time()
    for i, (train_images, train_labels) in enumerate(trainLoader):
        snn.zero_grad()
        optimizer.zero_grad()
        
        train_images = train_images.float().to(device)
        train_predicts = snn(train_images)
        train_labels = torch.zeros(train_labels.shape[0], num_classes).scatter_(1, train_labels.view(-1,1), 1.)
                                            #将index视作列（dim = 1），按照train_labels.view(-1,1)值作为index，把src =“1”插入到zeros中
        loss = criterion(train_predicts.cpu(), train_labels)
        train_loss = loss.item() + train_loss
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 25 == 0:#每100个batch报告一次，即100*100张图片报告一次
            print('Epoch [%d/%d], Train Step [%d/%d], Train Loss = %.5f' 
                    %(epoch + 1, num_epochs, i + 1, len(trainSet)//batch_size, train_loss))
            train_loss = 0
            print('Time elasped:', time.time() - start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    
    with torch.no_grad():
        for j, (test_images, test_targets) in enumerate(testLoader):
            optimizer.zero_grad()
            snn.zero_grad()
            
            test_images = test_images.float().to(device)
            test_predicts = snn(test_images)
            test_labels = torch.zeros(test_targets.shape[0], num_classes).scatter_(1, test_targets.view(-1,1), 1.)
            loss = criterion(test_predicts.cpu(), test_labels)
            
            _, predicted = test_predicts.cpu().max(1)
            total = total + float(test_targets.size(0))
            correct = correct + float(predicted.eq(test_targets).sum().item())
            if (j + 1) % 25 == 0:
                acc = 100. * float(correct)/float(total)
                print('Test Step [%d/%d], Acc: %.5f' %(j + 1, len(testLoader), acc))
    acc = float(100 * correct / total)
    print('Epoch [%d/%d] \t Test Accuracy Over Test Dataset: %.3f' %(epoch + 1, num_epochs, acc))
    acc_record.append(acc)
    if epoch % 5 == 0:
        print('Saving......')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('./history'):
            os.mkdir('./history')
        torch.save(state, './history/' + names + '.pk')



















