import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
            self,
            proposal_matcher,
            fg_bg_sampler,
            box_coder,
            cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        """根据预测值和真实值之间的IoU, 将预测值映射到真实值, 返回每个预测框对应的真实框
        :param proposal: 一张图片上预测出的所有roi, boxlist对象  N(num_anchors)x4
        :param target: 一张图片上的所有gt_box, boxlist对象  M(num_gt)x4

        :return matched_targets: boxlist对象, Nx4, 代表N个roi对应的gt_box
        """

        # 计算匹配质量矩阵(IoU), M(gt) x N(predict)
        match_quality_matrix = boxlist_iou(target, proposal)

        # N维tensor, 取值范围是0~M-1(gt index), -1(low~high), -2(<low)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")

        # matched_idxs有可能出现负值, 需要处理一下, 这里没有改变matched_idxs的值
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        return matched_targets

    def prepare_targets(self, proposals, targets):
        """计算真实的平移和缩放值
        :param proposals: 预测框 list[Boxlist]
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

        # 分别处理每张图片
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # BoxList对象, Nx4, 代表N个roi对应的gt_box
            matched_targets = self.match_targets_to_proposals(proposals_per_image, targets_per_image)
            # 当前图片上的每个roi对应的gt_box的label
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # 所有roi中属于背景的roi, 将其label置为0
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # low~high之间的roi, 将其label置为-1
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # 第一个参数指的是每个预测的anchor对应的gt_box, 第二个参数指的是预测的anchor
            # 返回值是RCNN论文中的 [t_x, t_y, t_w, t_h]
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        这个方法中, 首先根据IoU计算出每个roi对应的gt_box以及该gt_box的label
        之后从每张图片的所有roi中以固定的比例随机采样512个roi.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
            这两个参数第一个维度都是img_batch
        """

        # labels: fg,bg,discard     regression_targets: [t_x, t_y, t_w, t_h]
        labels, regression_targets = self.prepare_targets(proposals, targets)

        # 正负样本的mask, 这两个mask的维度与labels相同, 对于每张图片来说, 这两个mask中
        # 1的数量都是512(每张图片采样512个roi)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)

        # 为每张图片对应的BoxList对象添加 labels 和 regression_targets 属性
        for labels_per_image, regression_targets_per_image, proposals_per_image \
                in zip(labels, regression_targets, proposals):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in \
                enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 当前图片上的所有roi中, 经过随机采样之后保留的roi的索引值
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            # 当前图片上采样到的所有roi
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            # proposals中只保留采样到的样本
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals

        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)  # [num_roi, 81]
        box_regression = cat(box_regression, dim=0)  # [num_roi, 81*4]
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,  # 0.5
        # Overlap threshold for an RoI to be considered background
        # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,  # 0.5
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS  # (10., 10., 5., 5.)
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,  # 512
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION  # 0.25
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG  # False

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
