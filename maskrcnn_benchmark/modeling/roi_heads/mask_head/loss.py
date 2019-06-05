import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    在prepare_targets函数中被调用

    :param segmentation_masks: 一张图片上的SegmentationMask对象, 包含这张图片上所有生成
        的roi对应的gt_box的mask值
    :param proposals: BoxList对象, 代表一张图片上预测的所有roi
    :param discretization_size: 14

    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    masks = []
    M = discretization_size  # 14
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")

    # image size应该保持一致
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)

    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size  # 14

    def match_targets_to_proposals(self, proposal, target):
        """
        根据预测值和真实值之间的IoU, 将预测值映射到真实值, 返回每个预测框对应的真实框
        """

        # 计算匹配质量矩阵(IoU), M(gt) x N(predict)
        match_quality_matrix = boxlist_iou(target, proposal)

        # N维tensor, 取值范围是0~M-1(gt index), -1(low~high), -2(<low)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])

        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)

        return matched_targets

    def prepare_targets(self, proposals, targets):
        """
        :param proposals: list[BoxList] 每张图片上的roi, 只包含正样本
        """
        labels = []
        masks = []

        # 分别处理每张图片
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # BoxList对象, Nx4, 代表N个roi对应的gt_box
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            # 当前图片上的每个roi对应的gt_box的label
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # 由于训练阶段传入mask head的roi首先进行了负样本的过滤, 因此这里实际上都是
            # 正样本, 为了完整性保留这两行代码
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            # segmentation_masks是所有roi(正样本)对应的gt_box(根据IoU确定)的mask值
            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList]): 每张图片上的roi, 只包含正样本
            mask_logits (Tensor): [num_pos_roi, 81, 28, 28]
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION  # 14
    )

    return loss_evaluator
