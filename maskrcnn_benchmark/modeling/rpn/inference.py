import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from .utils import permute_and_flatten


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
            self,
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            min_size,
            box_coder=None,
            fpn_post_nms_top_n=None,
    ):
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def add_gt_proposals(self, proposals, targets):
        """
        对于 proposals 中每张图片的 boxlist, 直接将这张图片上的 gt_box 添加进去

        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList], [image1-si-boxlist, image2-si-boxlist, ...]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W

        返回值是一个 list, len(result)=batch_size, 每个元素都是一个 BoxList 对象
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # objectness的shape是[N,A,H,W], 现在要把每个A*H*W的特征图拉成一个向量, 如果直接进行
        # reshape操作, 展开的顺序是从A那一维开始的, 所以先交换维度再reshape, 先把H*W的特征图
        # 拉成一个向量, 再把所有特征图拼接起来
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)

        # rpn 中要进行的是不关心类别的二分类任务(object/bg)
        # [N, H*W*A]
        objectness = objectness.sigmoid()

        # [N, H*W*A, 4]
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        # 根据置信度选出前 k 个 anchors, k = pre_nms_top_n
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        # box_regression 中同样保留 topk 的anchors
        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]

        # boxList.bbox 返回对象中的 tensor, 将 batch 中所有图片的 anchors 拼接起来
        # boxList.bbox 是个二维的 tensor, 参考 anchor_generator.grid_anchors
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        # reshape 之后: [N, H*W*A, 4], 然后选出 topk
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        # 分别处理 batch 中的每一张图片
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)

            # 将超出图片边界的 anchors 进行裁剪
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # 将宽度或高度小于 min_size 的 anchors 移除
            boxlist = remove_small_boxes(boxlist, self.min_size)
            # nms
            boxlist = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]], 当前batch中所有level特征图上生成的anchors
                     anchors.shape: (batch_size, num_stages)
            objectness: list[tensor], 当前batch中每个level特征图的分类置信度, tensor四维
            box_regression: list[tensor], 当前batch中每个level特征图的box预测值

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        # 存放各个 level 处理过后的 anchors
        sampled_boxes = []

        num_levels = len(objectness)  # stage 的数量

        # anchors 第一个维度是 batch, 将其变成 stage(level)
        # [[image1-s1, image1-s2, ...], [image2-s1, image2-s2, ...], ...]
        # -> [[image1-s1, image2-s1, ...], [image1-s2, image2-s2, ...], ...]
        anchors = list(zip(*anchors))

        # 这步操作主要是进行超出图片anchors的处理,宽度高度为0的anchors的处理以及nms
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        # sampled_boxes 第一个维度是 level, 第二个维度是 batch_size, 将这两个维度置换
        boxlists = list(zip(*sampled_boxes))

        # 经过上面的置换之后, boxlists: [[img1-s1, img1-s2, ...], [img2-s1, img2-s2, ...], ...]
        # 将其转换成: [img1-boxlist, img2-boxlist, ...]
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        # 对于 fpn 来说, 上面进行 nms 时保留的 anchor 数量是对于每个 level 来说的
        # 这里对所有 level 的 anchor 再进行一次筛选, 选出 fpn_post_nms_top_n 个置信度较大的 anchor
        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        # [img1-boxlist, img2-boxlist, ...]
        return boxlists

    def select_over_all_levels(self, boxlists):
        """
        boxlists: [img1-boxlist, img2-boxlist, ...]
        """

        # batch_size
        num_images = len(boxlists)

        # TODO resolve this difference and make it consistent. It should be per image, and not per batch
        # 训练阶段的处理流程, 训练阶段是从一个 batch 的所有 image 中选出 post_nms_top_n 个 proposals
        if self.training:
            # 把一个 batch 中所有的 objectness 连接起来
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )

            # 每张图片的 boxlist 中 box 的数量
            box_sizes = [len(boxlist) for boxlist in boxlists]

            # fpn 中在 nms 之后要保留的 box 数量
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))

            # 从所有置信度中选出前 post_nms_top_n 个
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)

            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)  # 将索引值按照每张图片上box的数量进行划分

            for i in range(num_images):
                # BoxList 对象支持索引, 这里直接取出按置信度排序后的所有 box
                boxlists[i] = boxlists[i][inds_mask[i]]
        # 测试阶段的处理流程, 不同于训练阶段, 这时是从每张图片中选出 fpn_post_nms_top_n 个 proposals
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    """
    创建 RPNPostProcessor 对象
    """

    # fpn 所有 level 的输出最后保留的 boxes 数量
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN  # 2000
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST  # 2000

    # 对于 FPN 来说, 这里的参数指的是 per FPN level (not total)
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN  # 12000
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN  # 2000
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST  # 6000
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST  # 1000

    # 非极大值抑制, 0.7
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE  # 默认为 0
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector
