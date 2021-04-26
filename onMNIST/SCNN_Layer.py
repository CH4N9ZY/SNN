#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from Iterative_LIF_Model import mem_update, act_fun


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3), (32, 32, 1, 1, 3),]

# kernel size
cfg_kernel = [28, 14, 7]

# fc layer
cfg_fc = [128, 10]


# In[5]:


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# In[6]:


batch_size  = 100
learning_rate = 1e-3
decay = 0.2


# In[7]:


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes_0, out_planes_0, stride_0, padding_0, kernel_size_0 = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes_0, out_planes_0, kernel_size=kernel_size_0, stride=stride_0, padding=padding_0)
        in_planes_1, out_planes_1, stride_1, padding_1, kernel_size_1 = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes_1, out_planes_1, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike, decay)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike, decay)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike, decay)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike, decay)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs


# In[ ]:




