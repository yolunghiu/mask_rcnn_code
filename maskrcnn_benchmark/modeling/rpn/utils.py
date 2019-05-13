"""
Utility functions minipulating the prediction layers
"""

from ..utils import cat

import torch


def permute_and_flatten(layer, N, A, C, H, W):
    # layer: [N, A*C, H, W] --> [N, A, C, H, W]
    layer = layer.view(N, -1, C, H, W)

    # permute 相当于 np.transpose 函数
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, A, C]

    # 可以理解为: 当前特征图上共有 H*W*A 个 anchors, 每个 anchors 根据需求预测出 C 个值
    layer = layer.reshape(N, -1, C)  # [N, H*W*A, C]

    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    """将box_cls和box_regression由"按level组织"变为"按image组织"
    :param box_cls: 第一个维度是level
    :param box_regression: 第一个维度是level
    """
    box_cls_flattened = []
    box_regression_flattened = []

    # box_cls: [[num_img, num_anchors, H, W], ...]
    # box_regression: [[num_img, 4*num_anchors, H, W], ...]
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4  # num_anchors
        C = AxC // A  # 1

        # [N,AxC,H,W] --> [N,HxWxA,C] 即 [N,num_anchors,num_cls]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N,Ax4,H,W] --> [N,HxWxA,4] 即 [N,num_anchors,4]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    # [N,all_anchors,C] --> [N*all_anchors,C(1)]
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    # [N,all_anchors,4] --> [N*all_anchors,4]
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression
