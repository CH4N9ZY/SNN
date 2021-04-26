#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


thresh = 0.5 # neuronal threshold
lens = 1 # hyper-parameters of approximate function


# In[3]:


class ActFun(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)#把input变量保存到需要计算梯度的名单中（ctx不可写为其它的）
        return input.gt(thresh).float()#greater than thresh就会返回1，小于等于就会得到0
    
    @staticmethod
    def backward(ctx, grad_output):#grad_output是Loss对于ActFun的输出（y）的导数，这里要求Loss关于ActFun输入（x）的导数
                                   #就像Loss = (y_hat - y)^2, y = ActFun(x)，Loss关于ActFun输入的导数就等于Loss关于y的导数乘y关于x的导数
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens/2#因为正常的ActFun本身不可导（它是x超过threshold后从0直接跳跃到1的）
                                         #所以用这个函数近似计算在该处的导数，在threshold附近导数值为1，其他地方导数为0
        return grad_input * temp.float()*(1/lens)


# In[4]:


act_fun = ActFun.apply
decay = 0.2


# In[6]:


def mem_update(ops, x, mem, spike, decay):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike


# In[ ]:




