# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    该类主要实现了 BN 层的功能, 只不过其中的参数都是固定的, 而非可更新的
    具体原因是:
        1. batch size非常小
        2. 多GPU训练时, BN计算的statistics是单个GPU上的
        https://github.com/facebookresearch/maskrcnn-benchmark/issues/267
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
