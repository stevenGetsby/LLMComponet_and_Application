import torch
import torch.nn as nn
import numpy as np
import math

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.alpha = nn.Parameter(torch.ones(features))# 可训练参数 𝛼, 𝛽 ，因此会在model.parameter中
        self.beta = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x:torch.Tensor):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 返回Layer Norm的结果
        return self.alpha * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.beta