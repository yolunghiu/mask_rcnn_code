import torch


class Matcher(object):
    """
    这个类将预测的 box 与 gt_box 对应起来. 每个预测的 box 只有0个或1个匹配的 gt_box; 但
    一个 gt_box 可能对应于多个预测的 box

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """ 0.7, 0.3, True
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
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

        # match_quality_matrix is M (gt) x N (predicted)
        # 最大值, 最大值索引, N 维 tensor
        # 从每个预测的box出发,找到当前预测box与所有gt_box重叠最大的那个,不考虑阈值
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

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # 对每一个 gt_box 来说, 所有预测框中的最佳匹配值, M 维 tensor
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
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
