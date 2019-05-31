import torch
from torch import nn

from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor


class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()

        # 创建FPN2MLPFeatureExtractor对象, 对roi进行roialign, 并使用全连接层将池化后的特征图
        # 转换成特征向量
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)

        # 预测最终的分类置信度和box, out_channels是head中全连接层的神经元个数
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)

        # 创建PostProcessor对象,用于测试阶段对box进行过滤
        self.post_processor = make_roi_box_post_processor(cfg)

        # 创建FastRCNNLossComputation, 用来计算分类损失和回归损失
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): 多个level的特征图
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): [num_roi, 1024], 每个roi经过池化和fc层, 最终被转化成一个特征向量
            proposals (list[BoxList]): 训练阶段, 返回的是每张图片所有roi降采样之后保留的
                roi. 测试阶段, 返回的是每张图片上所有roi经过过滤之后保留的roi, coco中规定是
                100个.
            losses (dict[Tensor]): 训练阶段, 返回box head的loss. 测试阶段为空
        """

        # 在训练阶段, 从每张图片上以固定的正负样本比例采样512个roi(512是手动设置的参数)
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # pooler + heads, (num_rois, 1024)
        x = self.feature_extractor(features, proposals)

        # 最终预测的分类置信度和box预测值, [num_roi, 81], [num_roi, 81*4]
        class_logits, box_regression = self.predictor(x)

        # 测试阶段, 对box进行score的过滤, nms, 最后每张图片保留100个box
        if not self.training:
            # result是一个list, 每个元素对应一张图片, 是一个BoxList对象
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    return ROIBoxHead(cfg, in_channels)
