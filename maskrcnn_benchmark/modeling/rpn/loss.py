"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from .utils import concat_box_prediction_layers
from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        """根据预测值和真实值之间的IoU, 将预测值映射到真实值, 返回每个预测框对应的真实框
        :param anchor: 一张图片上的所有anchor, boxlist对象  N(num_anchors)x4
        :param target: 一张图片上的所有gt_box, boxlist对象  M(num_gt)x4

        :return matched_targets: boxlist对象, Nx4, 代表N个prediction anchor对应的gt_box
        """
        # 计算匹配质量矩阵(IoU), M(gt) x N(predict)
        match_quality_matrix = boxlist_iou(target, anchor)

        # N维tensor, 取值范围是0~M-1(gt index), -1, -2
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)

        # matched_idxs有可能出现负值, 需要处理一下
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        return matched_targets

    def prepare_targets(self, anchors, targets):
        """计算真实的平移和缩放值
        :param anchors: 预测框 list[Boxlist]
        :param targets: 真实框 list[Boxlist]
            上面两个参数维度都是img_batch
        
        :return:
            labels: fg,bg,discard list[Tensor] [img_batch, num_anchors]
            regression_targets: [t_x, t_y, t_w, t_h] list[Tensor]  [img_batch, num_anchors, 4]
        """
        # list中每个元素是一个N维tensor, 取值范围是[-1,0,1], -1代表discard, 0代表bg, 1代表fg
        labels = []

        # list中每个元素是Nx4的tensor, 代表[t_x, t_y, t_w, t_h], 实际的平移和缩放值
        regression_targets = []

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # 每个预测的anchor对应的gt_box    Nx4
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            # 每个预测的anchor对应的gt_box的label    N维
            matched_idxs = matched_targets.get_field("matched_idxs")

            # N维tensor, 只有0,1两个值, 1的位置代表该位置有匹配到的gt_box
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            # matched_idxs=-1 的位置为背景, 把label设置为0
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # 第一个参数指的是每个预测的anchor对于的gt_box, 第二个参数指的是预测的anchor
            # 返回值是RCNN论文中的 [t_x, t_y, t_w, t_h]
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[list[BoxList]]), 第一个维度是img_batch, 第二个维度是level
                每个level的anchor是一个BoxList对象
            objectness (list[Tensor]), 第一个维度是level
            box_regression (list[Tensor]), 第一个维度是level
            targets (list[BoxList]), 第一个维度是img_batch

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        # anchors: [num_imgs, (x)num_levels(个boxlist)] --> [num_imgs(个boxlist),]
        # 即将batch中每张图片各个level的boxlist对象合并成一个
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        # labels: fg,bg,discard  [img_batch, num_anchors]
        # regression_targets: t_x,t_y,t_w,t_h  [img_batch, num_anchors, 4]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        # 从所有预测值中随机采样一个batch的正负样本  [img_batch, num_anchors]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # 处理之前sampled_pos_inds和sampled_neg_inds: [img_batch, num_anchors], 可以看成二维矩阵
        #   矩阵中每一行都是0和1两种值, 1的位置代表采样到的样本数量, 这两个变量同一行中1的数量相加之后是
        #   batch_size_per_image, 即从同一张图片中采样batch_size_per_image个正负样本
        # 处理之后sampled_pos_inds和sampled_neg_inds: [all_sampled_inds], 处理过程是首先把img_batch
        #   展开, 展开后共有img_batch*num_anchors个数, 然后取出这些数中非0元素的索引值,
        #   取值范围是0~ img_batch*num_anchors-1, 后面对labels和regression_targets同样将img_batch展
        #   开, 这样就可以使用这两个索引变量直接进行取值
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        # [img_batch*batch_size_per_image]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        # objectness: [[num_img, num_anchors, H, W], ...] --> [img_batch*num_anchors, 1]
        # box_regression: [[num_img, 4*num_anchors, H, W], ...] --> [img_batch*num_anchors, 4]
        objectness, box_regression = \
            concat_box_prediction_layers(objectness, box_regression)

        # [img_batch*num_anchors]
        objectness = objectness.squeeze()

        # [img_batch, num_anchors] --> [img_batch*num_anchors]
        labels = torch.cat(labels, dim=0)
        # [img_batch, num_anchors, 4] --> [img_batch*num_anchors, 4]
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # sigmod结合交叉熵进行fg/bg二分类的损失函数
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    # 匹配到gt_box的anchor, 该位置设置为1, 其余位置(-1,-2)为0
    labels_per_image = matched_idxs >= 0

    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,  # 0.7
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,  # 0.3
        allow_low_quality_matches=True,
    )

    # 256, 0.5
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
