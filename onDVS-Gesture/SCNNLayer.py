#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from IterativeLIFModel import mem_update


# In[ ]:


# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(2, 32, 1, 1, 3), (32, 32, 1, 1, 3),]

# kernel size
cfg_kernel = [128, 64, 32]

# fc layer
cfg_fc = [128, 11]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class SCNN(nn.Module):
    
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes_0, out_planes_0, stride_0, padding_0, kernel_size_0 = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes_0, out_planes_0, kernel_size=kernel_size_0, stride=stride_0, padding=padding_0)
        in_planes_1, out_planes_1, stride_1, padding_1, kernel_size_1 = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes_1, out_planes_1, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        
    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(input.shape[0], cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(input.shape[0], cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(input.shape[0], cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(input.shape[0], cfg_fc[1], device=device)
        
        images_truple = torch.split(input, 1, len(input.shape)-4)
        
        for images in images_truple: # simulation time steps
            x = torch.squeeze(images)
            c1_received = self.conv1(x.float())
            c1_mem, c1_spike = mem_update(c1_received, c1_mem, c1_spike)
            x = F.avg_pool2d(c1_spike, 2)
            c2_received = self.conv2(x)
            c2_mem, c2_spike = mem_update(c2_received, c2_mem,c2_spike)
            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(input.shape[0], -1)
            h1_received = self.fc1(x)
            h1_mem, h1_spike = mem_update(h1_received, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_received = self.fc2(h1_spike)
            h2_mem, h2_spike = mem_update(h2_received, h2_mem,h2_spike)
            h2_sumspike += h2_spike
        outputs = h2_sumspike / len(images_truple)
        return outputs

