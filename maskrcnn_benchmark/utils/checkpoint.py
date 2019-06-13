import logging
import os

import torch

from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        """
        :param f: FPN中用到的配置是"catalog://ImageNetPretrained/MSRA/R-50"
        :return:
        """
        # 如果checkpoint记录文件已经存在, 获取最近一次保存的模型pth文件路径
        if self.has_checkpoint():
            f = self.get_checkpoint_file()

        # 如果指定的预训练模型不存在, 直接返回
        if not f:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        # 加载下载的预训练模型的参数
        checkpoint = self._load_file(f)
        # 将预训练模型的参数应用到当前模型当中
        self._load_model(checkpoint)

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """判断checkpoint记录文件是否存在"""
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """返回checkpoint文件中记录的最近一次保存的模型"""
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                # "log/model_0007500.pth"
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        """DetectronCheckpointer中这个方法被重写了"""
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        # 加载预训练模型参数之前, checkpoint字典中的'model'已经被弹出了
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
            self,
            cfg,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            save_to_disk=None,
            logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # f: catalog://ImageNetPretrained/MSRA/R-50
        if f.startswith("catalog://"):
            # 把paths_catalog.py作为一个module导入, 这个文件的路径是配置文件中设置的
            # 有可能存在于文件系统的任何位置, 因此不能直接import
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog",
                self.cfg.PATHS_CATALOG,  # paths_catalog.py的绝对路径
                True
            )

            # 'https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl'
            # 获取pretrained model的url
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f

        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f

        # 把pkl文件转换成Caffe2模型文件
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)

        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)

        return loaded
