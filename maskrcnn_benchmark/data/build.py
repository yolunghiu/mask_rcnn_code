import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_train, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    datasets = []

    # dataset_list:
    #   train: ("coco_2014_train", "coco_2014_valminusminival")
    #   test: ("coco_2014_minival",)
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)  # DatasetCatalog.get()

        # D <-> datasets, getattr(D, "COCODataset") <-> datasets.COCODataset
        factory = getattr(D, data["factory"])
        args = data["args"]  # dict, 包含root和ann_file两个键值对

        # for COCODataset, remove images without annotations during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms

        # **args: 'root', 'ann_file', 'remove_images_without_annotations', 'transforms'
        # **args指的是在函数调用时自动把dict中的键值对转换成指定参数名调用函数的方式, 这种方式的好处是
        #   参数名出现的顺序可以与函数参数定义时的顺序不一样
        dataset = factory(**args)  # COCODataset(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    """ 根据bins对x中的每个元素进行分组, 这里只有0和1两组
    :param x: aspect_ratios
    :param bins: [1]
    """
    bins = copy.copy(bins)
    bins = sorted(bins)

    # 对x中的每个元素应用lambda表达式
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))

    return quantized


def _compute_aspect_ratios(dataset):
    """计算Dataset中每张图片的高宽比(aspect ratio)"""
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    """
    :param sampler: 本代码上下文中, 传入的应该是RandomSampler对象
    :param aspect_grouping: [1]
    :param num_iters: 720000
    :param start_iter: 0
    """
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]

        # dataset中每张图片的高宽比
        aspect_ratios = _compute_aspect_ratios(dataset)

        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH  # 2
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)

        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER  # 720000
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH  # 1
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)

        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    # 默认为True
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )

    DatasetCatalog = paths_catalog.DatasetCatalog

    # ("coco_2014_train", "coco_2014_valminusminival") for train
    # ("coco_2014_minival",) for test
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    # transform中传入的target指的是image对应的BoxList对象
    transforms = build_transforms(cfg, is_train)
    # 创建COCODataset对象
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        # 实际创建的是torch.utils.data.sampler.RandomSampler
        sampler = make_data_sampler(dataset, shuffle, is_distributed)

        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS  # 4
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
