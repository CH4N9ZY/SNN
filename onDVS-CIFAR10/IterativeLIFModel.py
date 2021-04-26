#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


decay = 0.2
thresh = 0.5 # neuronal threshold
lens = 1 # hyper-parameters of approximate function


# In[5]:


def mem_update(x, mem, spike):
    '''
    spike是指上一时刻有没有firing，如果有的话就通过它将mem重置为0
    mem是指上一时刻的membrane potential，可以是大于threshold的，在计算这一时刻的时候重置就好了
    decay是用来描述如果没有接收到脉冲，membrane potential会自然回落的机制
    这个函数是用来计算这一时刻的membrane potential和是否spike的
    输入：
        接收到的信号（input），
        上一时刻未重置的membrane potential（mem），
        上一时刻是否spike（spike），
        自然回落的速度（decay）
    输出：
        这一时刻未重置的membrane potential（mem），
        这一时刻是否spike（spike）
    '''
    mem = decay * mem * (1. - spike) + x
    spike = act_func(mem)
    return mem, spike


# In[6]:


class act_func(torch.autograd.Function):
    '''
    因为这个函数需要自动求导，所以就要把它定义在autograd的类里面
    '''
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1/lens) * (abs(input - thresh) < lens/2).float()
        return grad_input


# In[ ]:


act_func = act_func.apply

