import torch


class Matcher(object):
    """
    这个类将预测的 box 与 gt_box 对应起来. 每个预测的 box 只有0个或1个匹配的 gt_box; 但
    一个 gt_box 可能对应于多个预测的 box.

    匹配操作基于MxN的矩阵match_quality_matrix进行, 矩阵中每个值代表(gt, predicted)的匹配度.
    例如对于基于IoU的box匹配来说, 矩阵中每个元素都代表了两个box之间的IoU.

    matcher对象返回的是一个N维tensor, 每个元素的值都是某个gt_box的索引值, 如果当前预测box没有
    与任何gt_box匹配, 则该位置为负数.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """ 0.7, 0.3, True
        Args:
            high_threshold (float): 置信度大于等于该阈值的 box 被选为候选框. 如 0.7
            low_threshold (float): 置信度小于high阈值但是大于等于low阈值的置为 BETWEEN_THRESHOLD,
                置信度小于low阈值的置为 BELOW_LOW_THRESHOLD
            allow_low_quality_matches (bool): 若为真, 则会产生额外一些只有低匹配度的候选框
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    # 实现这个函数之后, 该类的对象可以被当做函数调用, matcher(param)
    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): MxN 的矩阵,
            M 代表 gt_box 数量, N 代表预测的 box 数量

        Returns:
            matches (Tensor[int64]): N 维的 tensor, N[i] 是匹配到的 [0, M-1]
            范围内的 gt 或者一个负数(代表没匹配到任何 gt)
        """

        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # 从每个预测的box出发,为每个prediction找到匹配度最高的gt
        matched_vals, matches = match_quality_matrix.max(dim=0)

        # matches[0]: 第0个预测值与第matches[0]个gt_box匹配...
        # N 个预测 box 匹配到的 gt_box IoU 是0到1之间的任意值, 有可能 IoU 很低
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # 根据设置的阈值,将匹配到的低质量 box 的值置为负
        # 小于 0.3 的索引值
        below_low_threshold = matched_vals < self.low_threshold
        # 0.3~0.7 之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD  # -1
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS  # -2

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        # N维向量, 与prediction的box数量相同
        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        为那些仅具有低匹配度的predictions添加额外的matches. 具体来说, 就是给每一个gt找到一个具有
        最大IoU的prediction集合. 对于集合中的每一个prediction, 如果它还没有与其他gt匹配, 则把它
        匹配到具有最高匹配值的gt上 (注意是该prediction列中的最大值对于的gt, 并不一定是通过gt定位该
        预测值时的gt.
            TODO: 目前不太清楚为什么这么做, 猜测有可能是因为只想保留这个prediction, 因为它离那个
                选出的gt比较近, 但是这时候并不关心这个prediction的label到底设成了哪个gt
        ).
        """
        # 从gt_box出发, 对每一个 gt_box 来说, 所有预测框中的最佳匹配值, M 维 tensor
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # nonzero 函数返回的是 tensor 中非0元素的索引值
        # 这步操作得到的是一个 Mx2 的矩阵, 每一行中的两个元素代表 MxN 矩阵中的一个元素索引值
        gt_pred_pairs_of_highest_quality = \
            torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])

        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # MxN矩阵中每一行都可能出现多个最大值相等的情况, 因此会出现一个gt_box对应多个预测box的情况

        # 从gt出发选出的所有prediction的索引值
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]

        # 如果选出的prediction中预测值是low quality的(负值), 把它更新为
        # **与该prediction的IoU最大的那个gt**
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
