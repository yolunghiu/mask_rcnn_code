import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor
from .loss import make_rpn_loss_evaluator


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


# registry 指的是 maskrcnn_benchmark/modeling/registry.py 这个文件
# registry.RPN_HEADS 是文件中定义的一个 Registry 对象
# 这个装饰器的注解代码会在加载文件时被执行
@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()

        # 3x3 卷积过后特征图大小不变, 这里通道数也没变
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # 1x1 卷积之后输出的特征图通道数为 num_anchors, 因为输入特征图的每个 ceil 上都生成了
        # num_anchors 个 anchors, 这里将输出的特征图每个 ceil 上每个通道的值都作为当前 anchor 的分类置信度
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # 为每个 anchor 生成 4 个 bbox 坐标预测值
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            # 初始化参数为正态分布, 标准差 0.01(默认值为1, 为啥设置成这个值不清楚)
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        """
            x: [N, C, H, W]
            logits: [[N, num_anchors, H, W], ...] 有多少个level的特征图, list中就有多少元素
            bbox_reg: [[N, 4*num_anchors, H, W], ...] 有多少个level的特征图, list中就有多少元素
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels):
        # in_channels 是 backbone 网络结构输出的特征图通道数
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()

        # 创建AnchorGenerator对象, 用于在特征图上生成anchor
        anchor_generator = make_anchor_generator(cfg)

        # 获取RPNHead类的函数对象
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]

        # 创建RPNHead对象, in_channels 在 FPN 中为256, 也就是用来做检测的特征图的卷积核数量
        head = rpn_head(
            cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
        )

        # 其主要功能是将 bounding boxes 的表示形式编码成易于训练的形式(出自 R-CNN Appendix C)
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # RPNPostProcessor, 主要是进行 anchor 的筛选, nms 以及为 proposals 添加 gt_box (训练阶段)
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        # loss.py
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): 一个 batch 的 images
            features (list[Tensor]): 各个 level 的特征图, list 中的每个 Tensor 都是
                四维的 [N, C, H, W], 第一维是 batch_size
            targets (list[BoxList): 每张图片上的ground-truth boxes

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        # 根据传入的特征图, 使用RPNHead对象直接得到预测结果, 包括置信度和回归值
        # objectness 是置信度, 是一个 list, 第一个维度是 level, 第二个维度是一个 batch 中
        #   当前 level 输出的特征图   [[N, num_anchors, H, W], ...]
        # rpn_box_regression 中每个元素指的是每个 level 特征图的 box 预测值
        #   [[N, 4*num_anchors, H, W], ...]
        objectness, rpn_box_regression = self.head(features)

        # 在每张图片各个level的特征图上生成anchor, 每个boxlist都有个'visibility'属性
        #   用于标记有效的anchor
        # anchors.shape: (batch_size, num_stages)
        # [[boxlist, boxlist, ...], [boxlist, boxlist, ...], ...]
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:  # 默认为 False
            # 训练RPN-only模型时, 直接计算rpn loss即可, 无需转换anchors, 训练时用不到
            boxes = anchors
        else:
            # 对于end-to-end模型, 需要将anchors转换成boxes并采样成一个batch进行训练
            with torch.no_grad():
                # inference.py 移除不符合要求的anchor, 进行nms, 在所有anchor中保留
                # post_nms_top_n个anchor
                # TODO: 这个boxes变量之后是在哪里用到的?看到的时候回顾一下
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )

        # 计算rpn loss(前景/背景分类损失, box回归损失), 损失是通过从所有anchor中采样一个
        # batch的正负样本并计算这个batch中的损失
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]

        return boxes, {}


def build_rpn(cfg, in_channels):
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
