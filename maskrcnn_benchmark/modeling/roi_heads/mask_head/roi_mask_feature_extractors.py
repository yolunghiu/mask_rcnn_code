from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.poolers import Pooler
from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor

registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    先对特征图进行roialign, 池化成14x14大小, 然后使用4个卷积层提取特征
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION  # 14
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES  # 1/16
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO  # 0
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels  # 256 in FPN
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN  # False
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS  # (256, 256, 256, 256)
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION  # 1

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        # 这几个卷积层并没有改变特征图的尺寸
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        # MaskRCNNFPNFeatureExtractor
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
