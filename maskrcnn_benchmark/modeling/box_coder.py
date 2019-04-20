# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        根据 GT Boxes 将 proposals 编码, 用于 bounding box regression
        编码的结果是真实的平移和缩放值

        Arguments:
            reference_boxes (Tensor): ground truth boxes
            proposals (Tensor): boxes to be encoded, 这里指的是生成的 anchors
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE  # 宽度
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE  # 高度

        # 这里中心点坐标的计算与 anchor_generator 中的计算不太一样(宽度和高度没有减1)
        # 只要 proposal 与 gt 计算方式一样就不会影响最终结果
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        # 这四个变量对应于 RCNN 论文中的 [t_x, t_y, t_w, t_h]
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        根据 bounding box regression 的回归结果, 将回归之前预测的 proposals
        进行平移和缩放

        Arguments:
            rel_codes (Tensor): 平移和缩放值
            boxes (Tensor): proposals, 即生成的 anchors
        """

        boxes = boxes.to(rel_codes.dtype)

        # reference boxes 指的就是模型预测的 bbox regression 之前的 boxes
        # 论文中是 [P_x, P_y, P_w, P_h]
        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        # 每四个数采样一个数
        # d_* 是要学习的变换
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # pred_* 是根据学得的变换 d_*, 将回归之前的 boxes 进行变换的结果
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h

        # 在 encode 函数中,计算中心点坐标 x_ctr = x1 + 0.5*width, 而 width = x2 - x1 + 1
        # 所以 x2 = x_ctr + 0.5*width - 1
        # 也可以直接用 x1 + width - 1, 更好理解, -1 是因为坐标是像素级别的, 0代表第0个ceil
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
