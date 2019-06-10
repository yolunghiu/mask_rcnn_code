import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")

    positive_boxes = []
    positive_inds = []
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()

        # 创建MaskRCNNFPNFeatureExtractor对象, 对特征图进行roialign, 并用卷积层提取特征
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)

        # 创建MaskRCNNC4Predictor对象, 使用转置卷积和1x1卷积生成mask预测值
        self.predictor = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)

        # 创建MaskPostProcessor对象, 用于从所有类别的mask中选出概率值最大的类别
        self.post_processor = make_roi_mask_post_processor(cfg)

        # 创建MaskRCNNLossComputation对象, 用于计算mask loss
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): 默认情况下使用的是backbone网络提取的特征图
            proposals (list[BoxList]): 训练阶段是每张图片所有roi降采样之后保留的
                roi(512个). 测试阶段是每张图片上所有roi经过过滤之后保留的roi, coco中规定是100个.
            targets (list[BoxList], optional): 每张图片上的gt

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # 训练阶段只关心正样本
            all_proposals = proposals  # 正负样本都有, 共512个, 比例是1:4的比例
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        # 对特征图进行池化和特征提取
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            # [num_pos_roi, 256, 14, 14]
            x = self.feature_extractor(features, proposals)

        # [num_pos_roi, 81, 28, 28], 其中num_pos_roi是所有图片上正样本的数量
        mask_logits = self.predictor(x)

        # 测试阶段, 根据label从所有类别的mask中选出概率最大的类别的mask,
        # 并将其添加到BoxList对象的mask属性中
        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
