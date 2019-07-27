import itertools

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler


class GroupedBatchSampler(BatchSampler):
    """
    GroupedBatchSampler对另一个Sampler进行了包装.
    首先, GroupedBatchSampler保证所有样本按照分组策略进行分组之后, 采样时同一组别的样本应该
        出现在同一个batch当中.
    另外, 这个类尽可能保证元素采样的顺序与构造对象时传入的Sampler产生的顺序一致.

    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler, 代码中传入的是RandomSampler
        group_ids (list): 每个样本对应的group id
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    notes:
        Pytorch中的Sampler必须实现 __iter__ 和 __len__ 两个方法
        这个采样器的作用就是根据分组方法
    """

    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)  # dataset中每个样本对应的group id
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        # 这里只包含[0, 1]两个元素
        # sort(0)返回的第一个值是排好序的tensor, 第二个值是排好序的元素在原数组中的索引值
        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)

        # self.sampler返回的是这个sampler采样的样本在dataset中的索引值
        # [0, 2, 4, 1, 3]
        sampled_ids = torch.as_tensor(list(self.sampler))

        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        # order中每个位置的元素代表dataset中对应位置的样本的采样次序
        # 如: order[0]=1000, 代表第0个样本是第1000个被采样的样本
        # key: dataset中元素索引, value: sample order
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # mask的长度是数据集中所有图片的数量, mask中被采样到的元素为1, 未被采样为0
        mask = order >= 0

        # find the elements that belong to each individual cluster
        # list中是两个tensor, 每个tensor长度一样, 包含0和1两种元素
        clusters = [(self.group_ids == i) & mask for i in self.groups]

        # 每个组别内样本被采样的顺序
        relative_order = [order[cluster] for cluster in clusters]

        # s[0]=1000: s中第0个元素是第1000个被采样的样本
        # s.sort()[1], 对order进行排序, 取排序后对应顺序的索引
        # s[s.sort()[1]], 按order排好序的 FIXME 这他妈不就是s.sort()[0]
        permutation_ids = [s[s.sort()[1]] for s in relative_order]

        # s[0]=1000: s中第0个元素是第1000个被采样的样本
        # sampled_ids[1000]=10: 第1000个被采样的样本是dataset中的第10个样本
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches


    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)


    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)


if __name__ == '__main__':
    import numpy as np

    samp = RandomSampler(np.array([1, 3, 2, 5, 19]))
    print(list(samp))
