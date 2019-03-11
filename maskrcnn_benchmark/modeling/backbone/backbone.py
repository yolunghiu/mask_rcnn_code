# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet


@registry.BACKBONES.register(module_name="R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 256*4
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    # 创建ResNet基本骨架
    body = resnet.ResNet(cfg)

    # 256, 指的是从stage2输入的特征图的通道数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    # 256, 在FPN的配置文件中被重新赋值了, 默认是256*4
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(  # 这个函数返回的是一个函数对象 make_conv
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU  # 配置文件中默认都是False
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )

    # 将ResNet(body)和fpn进行组合
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels

    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )

    # cfg.MODEL.BACKBONE.CONV_BODY 默认为 R-50-C4
    # 若使用默认设置,实际返回的是 build_resnet_backbone(cfg)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
