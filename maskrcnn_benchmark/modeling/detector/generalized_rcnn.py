"""
Implements the Generalized R-CNN framework
"""

from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..roi_heads.roi_heads import build_roi_heads
from ..rpn.rpn import build_rpn


class GeneralizedRCNN(nn.Module):
    """
    该类主要包含以下三个部分:
    - backbone
    - rpn(option)
    - heads: 利用前面网络输出的 features 和 proposals 来计算 detections / masks.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        # backbone.py 创建 ResNet(resnet.py) 或 FPN(fpn.py) 骨架结构用于特征提取
        self.backbone = build_backbone(cfg)

        # rpn.py 构建 region proposal network
        # out_channels 在 ResNet 中为 256*4, 在 FPN 中为 256
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
        # proposal是经过筛选之后保留的post_nms_top_n个anchor, 是一个BoxList列表, 第一维度是img_batch
        # proposal_losses是二分类损失和box回归损失
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 这里调用roi_heads, 传入的是特征图, rpn预测的roi, gt_box
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
