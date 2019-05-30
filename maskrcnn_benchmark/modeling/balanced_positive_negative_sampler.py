import torch


class BalancedPositiveNegativeSampler(object):
    """
    该类用于生成采样 batch, 使得 batch 中的正负样本比例维持一个固定的数
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): 每张图片包含的样本个数(512)
            positive_fraction (float): 每个batch中正样本的比例
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched_idxs: list[Tensor], tensor中包含3类值: -1(ignored), 0(bg), 1(fg)
                [img_batch, num_anchors]
        Returns:
            pos_idx (list[tensor])  [img_batch, num_anchors]
            neg_idx (list[tensor])  [img_batch, num_anchors]
            维度与matched_idxs相同, 采样到的anchor被设置成了1, 未采样的anchor设置为0
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            # 一张图片上正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)

            # 一张图片上负样本的数量
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # 随机选取指定数量的正负样本(roi)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # 采样的正负样本在传入的matched_idxs中的索引值
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
