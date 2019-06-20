# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .distributed import DistributedSampler
from .grouped_batch_sampler import GroupedBatchSampler
from .iteration_based_batch_sampler import IterationBasedBatchSampler

__all__ = ["DistributedSampler", "GroupedBatchSampler", "IterationBasedBatchSampler"]


"""关于Pytorch中的Sampler

torch.utils.data.Sampler是Pytorch中所有Sampler的抽象类, 所有子类都必须重写 __iter__ 和 
__len__ 这两个方法. 第一个方法providing a way to iterate over indices of dataset elements, 
第二个方法returns the length of the returned iterators.

用for...in s语句实际上隐式调用了s中的特殊方法__iter__(), 拥有这一方法的序列称为可迭代对象, 
才可以应用for...in语句. 这个特殊方法返回了一个迭代器a, 迭代器a不断next()集合s中的值, 
当迭代器a next()完s中的所有元素后, 调用下一个next()将抛出StopIteration异常, 
这一异常就是for...in s语句遍历完毕的信号.

"""
