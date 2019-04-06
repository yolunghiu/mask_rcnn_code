# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


# 创建一个 GeneralizedRCNN 对象, 这样做可扩建性较强, 创建其他检测模型以同样的方式在这里指定即可
def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]  # GeneralizedRCNN
    return meta_arch(cfg)  # GeneralizedRCNN(cfg)
