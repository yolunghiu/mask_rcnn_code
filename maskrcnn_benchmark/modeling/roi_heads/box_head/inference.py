import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
            self,
            score_thresh=0.05,
            nms=0.5,
            detections_per_img=100,
            box_coder=None,
            cls_agnostic_bbox_reg=False
    ):
        super(PostProcessor, self).__init__()

        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
                每张图片保留self.detections_per_img个box
        """

        # [num_roi, 81], [num_roi, 81*4]
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # 每张图片的size
        image_shapes = [box.size for box in boxes]

        boxes_per_image = [len(box) for box in boxes]

        # 把所有图片上的box连起来  [num_roi, 4]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]

        # 根据预测的平移缩放值和相应的anchors, 把预测值转换成box坐标  [num_roi, 81*4]
        proposals = self.box_coder.decode(
            # [num_roi, 81*4], [num_roi, 4]
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        # 81
        num_classes = class_prob.shape[1]

        # 划分出每张图片的box坐标和分类置信度
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)

        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """将传入的boxes(tensor)构造成BoxList对象, 并为该对象添加scores属性
        :param boxes: [roi_per_img, 81*4]
        :param scores: [roi_per_img, 81]
        :param image_shape: 该boxlist对应的图片的尺寸
        """
        # [roi_per_img, 81*4] --> [roi_per_img*81, 4]
        boxes = boxes.reshape(-1, 4)
        # [roi_per_img, 81] --> [roi_per_img*81]
        scores = scores.reshape(-1)

        # 创建BoxList对象
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)

        return boxlist

    def filter_results(self, boxlist, num_classes):
        """首先移除 score<=score_thresh 的box, 然后对box进行nms, 最后对所有保留的box按score
        排序, 并保留置信度最大的前detections_per_img个box
        """
        # [roi_per_img*81, 4] --> [roi_per_img, 81*4]
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        # [roi_per_img*81] --> [roi_per_img, 81]
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device

        # result中存放每个类别对应的boxlist对象
        result = []

        # [roi_per_img, 81], scores中大于阈值的位置为1, 小于的为0
        inds_all = scores > self.score_thresh  # 0.05

        for j in range(1, num_classes):  # 0代表背景
            # 所有roi中, 第j个类别的score大于阈值的roi索引值
            inds = inds_all[:, j].nonzero().squeeze(1)

            # 第j类中, 所有大于阈值的score
            scores_j = scores[inds, j]

            # 第j类中, 所有score大于阈值的box
            boxes_j = boxes[inds, j * 4: (j + 1) * 4]

            # 构造当前类别的BoxList对象
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        # 将各个类别的BoxList对象合并成一个
        result = cat_boxlist(result)

        # 过滤之前共有 roi_per_img*81 个box, 经过上面score阈值过滤和nms后剩余的box数量
        number_of_detections = len(result)

        # self.detections_per_img默认为100, 即每张图片所有类别的box最多保留100个
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            # 返回第k小的元素和索引值
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
    # These are empirically chosen to approximately lead to unit variance targets
    # _C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS

    box_coder = BoxCoder(weights=bbox_reg_weights)

    # Only used on test mode
    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH  # 0.05

    nms_thresh = cfg.MODEL.ROI_HEADS.NMS  # 0.5

    # Maximum number of detections to return per image (100 is based on the limit
    # established for the COCO dataset)
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG  # 100
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG  # False

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
