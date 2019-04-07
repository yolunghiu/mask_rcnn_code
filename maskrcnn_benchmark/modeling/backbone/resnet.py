# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry

# ResNet 每个 stage 相关参数存储所用的数据结构
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # 当前 stage 索引值
        "block_count",  # 当前 stage 中残差块的数量
        "return_features",  # 布尔值, 代表当前 stage 最后一个残差块输出的特征图是否要返回
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# 只用 ResNet 不用 FPN 时,只有最后一个 stage 输出的特征图被用于下一步处理
# 使用 FPN 时,多个 stage 输出的特征图组成特征金字塔,用于物体检测
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # 根据配置文件中的定义,从namedtuple中取出对应的函数对象
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]  # StemWithFixedBatchNorm
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]  # R-50-C4
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]  # BottleneckWithFixedBatchNorm

        # 构建 stem module(也就是 resnet 的stage1, 或者 conv1) StemWithFixedBatchNorm(cfg)
        self.stem = stem_module(cfg)

        # ↓获取相应的信息来构建 resnet 的其他 stages 的卷积层↓

        # 当 num_groups=1 时为 ResNet, >1 时为 ResNeXt
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS  # 默认为1
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP  # 默认为64

        # in_channels 指的是 stem 的输出通道数, ResNet 论文中多种配置的网络都是64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        # ResNet 中每个 bottleneck 结构会先用 1x1 卷积将输入的特征图进行降维
        # 这个参数指的是 1x1 卷积核的数量
        stage2_bottleneck_channels = num_groups * width_per_group
        # 第二阶段的输出, resnet 系列标准模型可从 resnet 第二阶段的输出通道数判断后续的通道数
        # 默认为256, 则后续分别为512, 1024, 2048, 若为64, 则后续分别为128, 256, 512
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 默认为256

        # ↓根据stage2的参数,构建网络中的各个stage(其余stage的参数是以stage2为基准的)↓
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:  # len=4~5, (index, block_count, return_features)
            name = "layer" + str(stage_spec.index)

            # 计算每个stage的输出通道数, 每经过一个stage, 通道数都会加倍
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            # 计算输入特征图的通道数
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            # 计算输出特征图的通道数
            out_channels = stage2_out_channels * stage2_relative_factor

            # 当获取到所有需要的参数以后, 调用 `_make_stage` 函数,
            # 该函数可以根据传入的参数创建对应 stage 的模块
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,  # 当前stage中残差块的数量
                num_groups,  # ResNet时为1, ResNeXt时>1
                # Place the stride 2 conv on the 1x1 filter
                # Use True only for the original MSRA ResNet; use False for C2 and Torch models
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,  # default: True
                # 当处于 stage3~5时, 需要在开始的时候使用 stride=2 来downsize
                first_stride=int(stage_spec.index > 1) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features  # True or False

        # 根据配置文件的参数选择性的冻结某些层(requires_grad=False)
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)  # 2

    def _freeze_backbone(self, freeze_at):
        # 根据给定的参数冻结某些层的参数更新
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # resnet 的第一阶段, 即为 stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            # 将 m 中的所有参数置为不更新状态.
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            # 上面通过 self.add_module(name, module) 将各个stage都保存起来了
            x = getattr(self, stage_name)(x)
            # 如果该 stage 的 return_features 为 True,则将该module输出的特征图保存起来返回
            if self.return_features[stage_name]:
                outputs.append(x)

        # 将结果返回, outputs为列表形式, 元素为各个stage的特征图, 刚好作为 FPN 的输入
        return outputs


class ResNetHead(nn.Module):
    def __init__(
            self,
            block_module,
            stages,
            num_groups=1,
            width_per_group=64,
            stride_in_1x1=True,
            stride_init=None,
            res2_out_channels=256,
            dilation=1
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        # 根据给定的名称获取相应 block_module
        # "BottleneckWithFixedBatchNorm", "BottleneckWithGN"
        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
        transformation_module,
        in_channels,
        bottleneck_channels,
        out_channels,
        block_count,
        num_groups,
        stride_in_1x1,
        first_stride,
        dilation=1
):
    # 创建ResNet中的stage
    # block1: Bottleneck(in, bottleneck, out)
    # block2: Bottleneck(out, bottleneck, out)
    # ...

    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation
            )
        )

        # 当处于 stage3~5时, 需要在第一个 bottleneck 中使用 stride=2 来 downsample
        # 之后的 bottleneck 就不需要了
        stride = 1

        # 在一个 stage 中,只会在第一个 bottleneck 中出现 in_channels 不等于 out_channels 的情况
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    """
    创建 ResNet 中每个 stage 的 bottleneck 结构
    不同 stage 的 bottleneck block 的数量不同,对于 resnet50 来说,每一个阶段
    的 bottleneck block 的数量分别为 3,4,6,3,并且各个相邻 stage 之间的通道数都是两倍的关系
    """

    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups,
            stride_in_1x1,
            stride,
            dilation,
            norm_func  # 这个子类中会指定
    ):
        super(Bottleneck, self).__init__()

        # 处理 shortcut connection 时 x 和 最后 bottleneck 输出的特征图维度不一致问题
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample, ]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1  # reset to be 1

        # 在 resnet 原文中, 会在 conv3_1, conv4_1, conv5_1 处对输入的特征图在宽高上进行降采样
        # 降采样有两种策略: 在 bottleneck 的第一个 1x1 卷积中设置 stride=2 ,或在 3x3 卷积中设置 stride=2
        # stride_in_1x1=True 时,使用第一种策略
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation
        )
        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv2, self.conv3, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out


class BaseStem(nn.Module):
    """
    该类负责构建 ResNet 的 stem 模块
    在 ResNet_50 中,该阶段主要包含一个 7×7 大小的卷积核
    在 MaskrcnnBenchmark 的实现中,为了可以方便的复用实现各个 stage 的代码,
    它将第二阶段最开始的 3×3 的 max pooling 层也放到了 stem 中的 forward 函数
    """

    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS  # 64

        # 输入的 channels 为 3, 输出为 64
        self.conv1 = Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
            self,
            in_channels,
            bottleneck_channels,
            out_channels,
            num_groups=1,
            stride_in_1x1=True,
            stride=1,
            dilation=1
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
