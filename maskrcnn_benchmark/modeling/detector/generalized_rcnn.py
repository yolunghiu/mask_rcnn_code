# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    # 该类是 MaskrcnnBenchmark 中所有模型的共同抽象, 目前支持 boxes 和 masks 两种形式的标签
    # 该类主要包含以下三个部分:
    # - backbone
    # - rpn(option)
    # - heads: 利用前面网络输出的 features 和 proposals 来计算 detections / masks.

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        # backbone.py 创建ResNet(resnet.py)或FPN(fpn.py)骨架结构用于特征提取
        self.backbone = build_backbone(cfg)
        # rpn.py
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 自定义的数据结构,将不同尺寸的图片填充成相同尺寸并保存原始尺寸信息
        images = to_image_list(images)
        # 利用 backbone 网络获取图片的 features
        features = self.backbone(images.tensors)
        # 利用 rpn 网络获取 proposals 和相应的 loss
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # 如果是训练阶段,返回所有loss
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        # 如果是测试阶段,返回检测结果
        return result
