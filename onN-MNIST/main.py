#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ILIFModel import *
from SCNN_Layer import *
import os
import time


# In[2]:


def read2Dspikes(filename):
    '''
    Reads two dimensional binary spike file and returns a TD event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.
    
    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in microseconds.
    Arguments:
        * ``filename`` (``string``): path to the binary file.
    '''
    with open(filename, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = np.asarray([x for x in inputByteArray])
    xEvent =   inputAsInt[0::5]
    yEvent =   inputAsInt[1::5]
    pEvent =   inputAsInt[2::5] >> 7
    tEvent =( (inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5]) ) & 0x7FFFFF
    return event(xEvent, yEvent, pEvent, tEvent/1000) # convert spike times to ms


# In[3]:


class event():
    '''
    This class provides a way to store, read, write and visualize spike event.
    Members:
        * ``x`` (numpy ``int`` array): `x` index of spike event.
        * ``y`` (numpy ``int`` array): `y` index of spike event
                                        (not used if the spatial dimension is 1).
        * ``p`` (numpy ``int`` array): `polarity` or `channel` index of spike event.
        * ``t`` (numpy ``double`` array): `timestamp` of spike event.
                                        Time is assumend to be in ms.
    '''
    def __init__(self, xEvent, yEvent, pEvent, tEvent):
        if yEvent is None:
            self.dim = 1
        else:
            self.dim = 2
        
        self.x = xEvent if type(xEvent) is np.ndarray else np.asarray(xEvent)
        # x spatial dimension
        self.y = yEvent if type(yEvent) is np.ndarray else np.asarray(yEvent)
        # y spatial dimension
        self.p = pEvent if type(pEvent) is np.ndarray else np.asarray(pEvent)
        # spike polarity
        self.t = tEvent if type(tEvent) is np.ndarray else np.asarray(tEvent)
        # time stamp in ms
        
        if not issubclass(self.x.dtype.type, np.integer):
            self.x = self.x.astype('int')
        if not issubclass(self.p.dtype.type, np.integer):
            self.p = self.p.astype('int')
        if self.dim == 2:
            if not issubclass(self.y.dtype.type, np.integer):
                self.y = self.y.astype('int')
        
        self.p -= self.p.min()
    
    def toSpikeTensor(self, emptyTensor, samplingTime=1, randomShift=False, binningMode='OR'):
        # Sampling time in ms
        '''
        Returns a numpy tensor that contains the spike events sampled in bins of `samplingTime`.
        The tensor is of dimension (channels, height, width, time) or``CHWT``.
        Arguments:
            * ``emptyTensor`` (``numpy or torch tensor``): an empty tensor to hold spike data 
            * ``samplingTime``: the width of time bin to use.
            * ``randomShift``: flag to shift the sample in time or not. Default: False.
            * ``binningMode``: the way spikes are binned. 'SUM' or 'OR' are supported. Default: 'OR'
        '''
        if randomShift is True:
            tSt = np.random.randint(
                max(
                    int(self.t.min() / samplingTime),
                    int(self.t.max() / samplingTime) - emptyTensor.shape[3],
                    emptyTensor.shape[3] - int(self.t.max() / samplingTime),
                    1,
                )
            )
        else:
            tSt = 0
        
        xEvent = np.round(self.x).astype(int)
        pEvent = np.round(self.p).astype(int)
        tEvent = np.round(self.t/samplingTime).astype(int) - tSt
        
        if self.dim == 1:
            validInd = np.argwhere( (xEvent < emptyTensor.shape[2]) &
                                    (pEvent < emptyTensor.shape[0]) &
                                    (tEvent < emptyTensor.shape[3]) &
                                    (xEvent >= 0) &
                                    (pEvent >= 0) &
                                    (tEvent >= 0))
            if binningMode.upper() == 'OR':
                emptyTensor[pEvent[validInd],
                            0, 
                            xEvent[validInd],
                            tEvent[validInd]] = 1/samplingTime
            elif binningMode.upper() == 'SUM':
                emptyTensor[pEvent[validInd],
                            0, 
                            xEvent[validInd],
                            tEvent[validInd]] += 1/samplingTime
            else:
                raise Exception('Unsupported binningMode. It was {}'.format(binningMode))

        elif self.dim == 2:
            yEvent = np.round(self.y).astype(int)
            validInd = np.argwhere((xEvent < emptyTensor.shape[2]) &
                                   (yEvent < emptyTensor.shape[1]) & 
                                   (pEvent < emptyTensor.shape[0]) &
                                   (tEvent < emptyTensor.shape[3]) &
                                   (xEvent >= 0) &
                                   (yEvent >= 0) & 
                                   (pEvent >= 0) &
                                   (tEvent >= 0))

            if binningMode.upper() == 'OR':
                emptyTensor[pEvent[validInd], 
                            yEvent[validInd],
                            xEvent[validInd],
                            tEvent[validInd]] = 1/samplingTime
            elif binningMode.upper() == 'SUM':
                emptyTensor[pEvent[validInd], 
                            yEvent[validInd],
                            xEvent[validInd],
                            tEvent[validInd]] += 1/samplingTime
            else:
                raise Exception('Unsupported binningMode. It was {}'.format(binningMode))
            
        return emptyTensor


# In[4]:


class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        fileIndex  = self.samples[index, 0]
        label  = torch.tensor(self.samples[index, 1], dtype=torch.int64)

        data = read2Dspikes(
                            self.path + str(fileIndex.item()) + '.bs2'
                        ).toSpikeTensor(torch.zeros((2,34,34,300)),samplingTime=1.0)
        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        #return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel
        return data, label
    
    def __len__(self):
        return self.samples.shape[0]


# In[2]:


datasetPath = '../../../DATA/N-MNIST/slayerPytorch-master/example/02_NMNIST_MLP/NMNISTsmall/'
sampleFile = '../../../DATA/N-MNIST/slayerPytorch-master/example/02_NMNIST_MLP/NMNISTsmall/'
train_index = 'train1K.txt'
test_index = 'test100.txt'
samplingTime = 1.0
sampleLength = 300


# In[6]:


trainingSet = nmnistDataset(datasetPath = datasetPath, 
                            sampleFile  = sampleFile + train_index,
                            samplingTime= samplingTime,
                            sampleLength= sampleLength)
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers = 0)


# In[7]:


testSet = nmnistDataset(datasetPath = datasetPath, 
                            sampleFile  = sampleFile + test_index,
                            samplingTime= samplingTime,
                            sampleLength= sampleLength)
testLoader = DataLoader(dataset=testSet, batch_size=8, shuffle=False, num_workers = 0)


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


device


# In[10]:


start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = []
num_epochs = 100 # max epoch
num_classes = 10
batch_size = 8
names = 'STBPmodelN-MNIST'


# In[11]:


learning_rate = 1e-3


# In[12]:


snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)


# In[13]:


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# In[14]:


for epoch in range(num_epochs):
    train_loss = 0
    start_time = time.time()
    for i, (train_images, train_labels) in enumerate(trainLoader):#共计125个batch
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
                    %(epoch + 1, num_epochs, i + 1, len(trainingSet)//batch_size, train_loss))
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


# In[ ]:




