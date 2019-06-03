from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Predictor")
class MaskRCNNC4Predictor(nn.Module):
    """
    这个结构是mask rcnn论文中fig4/right展示的结构, 输入的特征图尺寸是14,
    这里使用一个转置卷积+1x1卷积, 输出尺寸为28的特征图
    """

    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES  # 81
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]  # 256
        num_inputs = in_channels  # 256

        # 转置卷积, 上采样两倍, 14-->28
        self.conv5_mask = ConvTranspose2d(in_channels=num_inputs, out_channels=dim_reduced,
                                          kernel_size=2, stride=2, padding=0)
        # 1x1卷积
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


@registry.ROI_MASK_PREDICTOR.register("MaskRCNNConv1x1Predictor")
class MaskRCNNConv1x1Predictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNConv1x1Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES  # 81
        num_inputs = in_channels  # 256

        # 1x1卷积, 只改变了特征图的通道数, 通道数设置为类别数, 即每个类别预测一个mask
        self.mask_fcn_logits = Conv2d(num_inputs, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        :param x: 对backbone的输出进行特征提取之后的特征图
        :return mask_logits: 使用1x1卷积对 每个类别预测一个mask
        """
        return self.mask_fcn_logits(x)


def make_roi_mask_predictor(cfg, in_channels):
    # MaskRCNNC4Predictor
    func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]

    return func(cfg, in_channels)
