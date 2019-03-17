# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


class BufferList(nn.Module):
    """
    把创建对象时传进来的 buffers 添加到 self._buffers 中, 键是该 buffer 在 Module 中的序号
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        # self._buffer 是一个 OrderedDict
        return len(self._buffers)

    def __iter__(self):
        # self._buffers.values() 返回字典(键值对)中所有的值
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
            self,
            sizes=(128, 256, 512),
            aspect_ratios=(0.5, 1.0, 2.0),
            anchor_strides=(8, 16, 32),
            straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        # 生成每个 stage 的特征图上第一个 ceil 上的所有 anchors
        if len(anchor_strides) == 1:  # faster rcnn
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:  # fpn
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]

        # 各个 stage 的特征图对应的 stride
        self.strides = anchor_strides
        # 各个 stage 的特征图对应的 anchors
        self.cell_anchors = BufferList(cell_anchors)
        # 对于超出图片的 anchor, 这个值为 0 时直接移除此 anchor, -1 或 10000 则裁剪 anchor
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        # 返回每个 stage 的特征图上的每个 ceil 上有几个 anchor
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
                grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                    (anchors[..., 0] >= -self.straddle_thresh)
                    & (anchors[..., 1] >= -self.straddle_thresh)
                    & (anchors[..., 2] < image_width + self.straddle_thresh)
                    & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors


# faster rcnn 与 fpn 中生成 AnchorGenerator 对象
def make_anchor_generator(config):
    # (32, 64, 128, 256, 512)
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES

    # (0.5, 1.0, 2.0)
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS

    # 对于 ResNet-C4 是 16, 对于 FPN 是 (4, 8, 16, 32, 64)
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE

    # 对于超出图片的 anchor 如何处理, 这个值为 0 时直接移除此 anchor, -1 或 10000 代表裁剪 anchor
    # Faster-RCNN 论文中在训练阶段直接移除跨边界的 anchor, 在测试阶段对跨边界的 anchor 进行裁剪
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        # FPN 中 P2~P6 依次使用不同的 stride, 每层特征图上只有一种 size 的 anchor
        assert len(anchor_stride) == len(anchor_sizes), \
            "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        # Faster RCNN 中只用了一个 stage 输出的特征图, 在这个特征图上设置不同 size 的 anchors
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"

    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)

    return anchor_generator


# 用与 retinanet
def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])

# 上面生成了 9 个 anchor, 应该是使用了 3 中 scales 和 3 中 ars, 这 9 个 anchors 应该都是
# 特征图上第一个点上的 anchor, 只要这个位置的 anchor 坐标确定了, 其他位置的 anchor 就能根据 stride 算出来


def generate_anchors(
        stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    # base anchor
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1

    # 基于 base anchor 生成的 中心点坐标以及面积相同 但是 高宽比 不同的 anchors
    anchors = _ratio_enum(anchor, aspect_ratios)

    anchors = np.vstack(
        # [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )

    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """传入 anchor 的两个坐标, 返回这个 anchor 的 宽,高,中心点横坐标,中心点纵坐标"""
    w = anchor[2] - anchor[0] + 1  # x2 - x1 + 1
    h = anchor[3] - anchor[1] + 1  # y2 - y1 + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """将 (w,h,x_ctr,y_ctr) 的 anchor 转换成 (x1,y1,x2,y2) 的形式"""
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """根据给定的 anchor 以及相对这个 anchor 的高宽比, 计算出面积相同但高宽比不同的多个 anchors"""
    # 计算 base_anchor 的相关信息
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # base_anchor 的面积
    size = w * h

    # 根据高宽比获取 size_ratios 变量, 后续会用该变量对 box 的高宽比进行转化
    size_ratios = size / ratios

    # ws = sqrt(size) / sqrt(ratios)
    # hs = sqrt(size) * sqrt(ratios)
    # 高宽比 = hs/ws = sqrt(ratios) * sqrt(ratios) = ratios
    # round 代表四舍五入, 默认只保留整数部分
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)

    # 根据新的 w 和 h, 生成新的 box 坐标(x1, x2, y1, y2) 并将其返回
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors


def _scale_enum(anchor, scales):
    """将给定的 anchor 按照 scales 进行缩放"""
    # 获取 anchor 的宽, 高, 以及中心坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # 将宽和高各放大 scales 倍
    # 这里的倍数是每个特征图上设置的 anchor_size 除以该特征图对应 stride 的结果
    ws = w * scales
    hs = h * scales

    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors


if __name__ == '__main__':
    anchors = generate_anchors()
    print(anchors)
