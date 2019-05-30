import torch
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign
from .utils import cat


class LevelMapper(object):
    """根据FPN论文中的公式计算每个RoI属于哪个FPN feature map level
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min  # 2
        self.k_max = k_max  # 5
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        :param boxlists (list[BoxList])
        :return target_lvls (Tensor): 一维tensor, 标记每个box输入哪个level
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        # 根据这个公式计算每个RoI的FPN Level
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)

        # 从0开始, 最大是3
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """对roi进行roialign操作
    输入各个level的特征图和每张图片上预测的所有roi(经过筛选之后的), 首先使用LevelMapper
    计算每个roi应该对应到哪个level的特征图上, 然后分别处理各个level上的roi, 对其进行roialign
    操作.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """ 7 (0.25, 0.125, 0.0625, 0.03125) 2
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size

        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()  # 2
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()  # 5
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        """把要进行池化操作的box进行格式转换
        boxes (list[BoxList]), 第一个维度是img_batch
        """
        # Nx4, 将一个batch中所有图片上的box连起来
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype

        # [0, 0, ..., 1, 1, ..., 2, 2, ..., img_batch-1], Nx1
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )

        # Nx5, [    [img_id, x1, y1, x2, y2],
        #           [img_id, x1, y1, x2, y2],
        #           ...
        #      ]
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois

    def forward(self, x, boxes):
        """
        :param x (list[Tensor]): 各个level的特征图
        :param boxes (list[BoxList]): 要进行池化操作的boxes, 第一个维度是img_batch
        :return result (Tensor[N, 256, 7, 7]) N指的是所有roi
        """
        num_levels = len(self.poolers)

        # Nx5, 在每个box之前添加了对应image的索引值
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        # 一维tensor, 长度为所有box的数量, rois和levels的索引值是一一对应的
        # 这个level指的是每个预测的roi根据area来映射到不同level的特征图进行后续处理
        levels = self.map_levels(boxes)

        num_rois = len(rois)  # N
        num_channels = x[0].shape[1]  # 256
        output_size = self.output_size[0]  # 7

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros((num_rois, num_channels, output_size, output_size),
                             dtype=dtype, device=device)

        # level是当前特征图的level, pooler是ROIAlign对象
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            # 所有proposal根据area计算出应该属于哪个FPN level, 这里找出所有
            # 计算结果中属于当前level特征图的proposal
            idx_in_level = torch.nonzero(levels == level).squeeze(1)

            # 属于当前level的roi
            rois_per_level = rois[idx_in_level]

            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
