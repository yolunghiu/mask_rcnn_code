# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


# 在 backbone.py 的 build_resnet_fpn_backbone() 中被调用
class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()

        self.inner_blocks = []
        self.layer_blocks = []

        # 假设我们使用的是 ResNet-50-FPN 和配置, 则 in_channels_list 的值为:
        # [256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, start=1):  # idx从1开始
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue

            # 1x1 卷积对每个stage最后输出的特征图进行通道降维, inner_block_module
            inner_block_module = conv_block(in_channels, out_channels, 1)
            # 3x3 卷积对 (lateral + top-down) 之后的特征图进行特征提取, layer_block_module
            # P5 由于没有 top-down 结构, 直接对 lateral进行 3x3 卷积
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)

            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)

            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # 先计算最后一层特征图 (C5) 经过 1x1 卷积之后的结果
        # 因为 top-down pathway 中底层特征图是上层特征图上采样叠加旁路连接得来的
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])

        # 存储 P2~P5
        results = []

        # P5 就是将 C5 进行 1x1 卷积之后在做 3x3 卷积得到的特征图, 因为 P5 上层没有特征图
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))  # P5

        # 计算 P4~P2
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue

            # 两倍上采样 P_n+1, 直接进行最近邻采样
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            # 对 C_n 进行 1x1 卷积
            inner_lateral = getattr(self, inner_block)(feature)

            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)

            # 对 lateral 和 top-down connection 进行 element-wise addition
            last_inner = inner_lateral + inner_top_down
            # 对相加后的特征图进行 3x3 卷积操作, 解决上采样的混叠效应
            # results 列表存储特征图的顺序是 [P2, P3, P4, P5]
            results.insert(0, getattr(self, layer_block)(last_inner))

        # RetinaNet 中用到的 op
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        # FPN 中用到的 op
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)  # P6

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    # 最后一级的 max pool 层
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
